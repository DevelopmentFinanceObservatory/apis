"""Assemble response parts and render confirmation sentences without LLM or graph calls."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

from code_index.shared import get_measure_index
from dfo_platform import Cause
from dfo_platform.roles import FlowFamily
from search.engine.closest_match import spec_closest_match_fact, substituted_readings_fact
from search.engine.config import settings
from search.engine.departures import (
    NO_ENTITY_WORDS,
    SOURCE_REF,
    merged_entity_facts,
    scope_limited_fact,
    spec_departure_facts,
    stated_facts,
    without_request_words,
)
from search.engine.qualifier_match import _normalize
from search.engine.region.codes import (
    ALTERNATES_UNEVALUATED,
    ELECTED_SHAPE_NO_DATA,
    NEAR_NEIGHBOR_AMBIGUOUS,
    RETRIEVAL_SCORE_WEAK,
)
from search.engine.region.near_neighbor import NEAR_NEIGHBOR_UNRESOLVED_MESSAGE
from search.engine.retrieval.bound_values import identity_names_measure_family
from search.engine.skills import match_spec_skills
from search.engine.spec_build import capping_stat_var_items
from search.heading.departure_fact import DepartureFact, ordered_facts
from search.models import (
    _UNRESOLVED_NO_DATA_REASONS,
    UNIT_REF,
    AmbiguousEntityGroup,
    CandidateSet,
    CandidatesPart,
    DefinitePart,
    Departure,
    DepartureCode,
    EntityRoleKey,
    NoData,
    NoDataGate,
    NoDataPart,
    NoDataReason,
    Operation,
    Ordering,
    ParsedInput,
    ParsedPart,
    PartRestatementReason,
    ResolvedQualifier,
    RoleAxis,
    SentenceToken,
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
    alternate_preserves_bound_role,
    no_data_message,
    no_data_sentence_plan,
)
from search.render.plan import render_sentence_plan
from search.render.tokens import plan_words, sentence_from_plan
from skills_index import SkillsIndex

if TYPE_CHECKING:
    from search.engine.region.result import RegionResult
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

# Cap for the resolved.stat_vars.items of a candidate whose identity names a measure
# family. A candidates part repeats every candidate's members, and an ODA-family item is
# about 1 KB, so six candidates at the definite cap of 2,000 would carry about 12 MB. At
# 200 each, they carry about 1.2 MB. A resubmit of such a candidate re-elects its members
# from the code index and ignores the posted list. A standard-shape candidate keeps the
# definite cap, since its items are the membership a resubmit posts back.
CANDIDATE_STAT_VARS_MAX = 200


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


def _shown_source(specs: Sequence[Spec]) -> str | None:
    """The source the bylines of `specs` name when they name one alone, else `None`."""
    sources = {
        source.text
        for spec in specs
        if (source := _heading(spec).byline.source) is not None and source.text
    }
    return sources.pop() if len(sources) == 1 else None


def _with_source_shown(
    facts: Sequence[DepartureFact], *, shown: Sequence[Spec]
) -> list[DepartureFact]:
    """`facts`, with each dataset the answer does not read (the source token's ref) stating
    as `shown` the source the part's bylines name, or `None` when its readings come from
    several."""
    source = _shown_source(shown) if any(f.binding_ref == SOURCE_REF for f in facts) else None
    return [
        dataclasses.replace(fact, shown=source) if fact.binding_ref == SOURCE_REF else fact
        for fact in facts
    ]


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
    basis as `shown` on a price-basis fact that lacks it, the byline's source on a dataset
    fact, and the `closest_match` fact of its reading (`_with_closest_match`)."""
    facts = [
        _with_price_basis_shown(fact, unit_text=heading.byline.unit_text or "")
        for fact in part_departure_facts(
            spec, raised=raised, operations=operations, entity_words=entity_words
        )
    ]
    return _with_closest_match(
        _with_source_shown(facts, shown=[spec]),
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


def restatement_reason_for(
    restatement: ParsedPart | None,
    *,
    reason: PartRestatementReason | None = None,
) -> PartRestatementReason | None:
    """Return the wire reason paired with a part's restatement.

    Exactly one of ``restatement`` and the reason is set. A present restatement yields
    ``None``. Otherwise use the supplied reason, or
    ``no_phrase_inputs`` when no phrase existed to echo.
    """
    if restatement is not None:
        return None
    return reason if reason is not None else "no_phrase_inputs"


def build_definite_part(
    spec: Spec,
    *,
    variable_text: str | None,
    warnings: list[Warning],
    timing_ms: Timing | None = None,
    include_sentence: bool = False,
    restatement: ParsedPart | None = None,
    restatement_reason: PartRestatementReason | None = None,
    operations: Sequence[Operation] = (),
    include_skills: bool = False,
    skills_index: SkillsIndex | None = None,
    departure_facts: Sequence[DepartureFact] = (),
    entity_words: Mapping[str, str] | None = NO_ENTITY_WORDS,
) -> DefinitePart:
    """Build a definite part around its answering spec.

    Every definite spec must contain a StatVar. The part's election warnings
    (`slot_defaulted`, `family_total_absent`) stay on the part. Skill matches attach when
    requested and indexed. ``restatement_reason`` is normalised against this part's own
    ``restatement``. ``departure_facts`` are the raised facts whose warnings the part or
    its envelope holds. ``definite_departures`` adds those the spec, the operations
    and the reading's match carry. ``entity_words`` are the request's own words for each
    entity name (`RegionResult.entity_words`), in which a group read as members is asked.
    """
    # A definite spec must carry at least one confirmed StatVar.
    assert spec.resolved.stat_vars.items, "definite Spec has empty stat_vars"
    if include_sentence:
        spec = with_sentence(spec)
    if include_skills and skills_index is not None:
        spec = with_skills(spec, index=skills_index)
    return DefinitePart(
        status="definite",
        variable_text=variable_text,
        warnings=warnings,
        timing_ms=timing_ms,
        spec=spec,
        restatement=restatement,
        restatement_reason=restatement_reason_for(restatement, reason=restatement_reason),
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
    timing_ms: Timing | None = None,
    include_sentence: bool = False,
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
    qualifier_terms: list[ResolvedQualifier] | None = None,
    unit_alternatives: list[UnitAlternative] | None = None,
    unit_alternatives_truncated: bool = False,
    restatement: ParsedPart | None = None,
    restatement_reason: PartRestatementReason | None = None,
    operations: Sequence[Operation] = (),
    include_skills: bool = False,
    skills_index: SkillsIndex | None = None,
    cause: Cause | None = None,
    window_gap_fact: DepartureFact | None = None,
    departure_facts: Sequence[DepartureFact] = (),
    request_words: bool = True,
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
    ``entity_ambiguous``. ``qualifier_terms`` is used for ``qualifier_not_bound``.
    ``unit_alternatives`` and ``unit_alternatives_truncated`` carry a
    ``unit_not_resolved`` refusal's capped currencies. ``None`` omits both.
    ``NoData.message`` is the bare no-data sentence with every term the part names, set
    whether or not a sentence is requested.

    Sentence plans reuse each near-match's tokens under a gate-specific lead-in. A skills
    request adds ``skills_not_matchable`` for the answer and matches available near-matches
    when an index is available. Restatement fields are normalised together. ``cause`` is
    populated for ``rate_coverage_gap``. ``window_gap_fact`` is the `year_unavailable` fact of
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
        # out. Only what Search itself does not do departs.
        facts = tuple(fact for fact in facts if fact.code in _SEARCH_DEPARTURE_CODES)
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
    if include_sentence:
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
            departures=departures,
        )
        rendered = sentence_from_plan(plan)
    else:
        rendered = None
        plan = None
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
            ),
            nearest_real=nearest_real,
            flagged_match=flagged_match,
            gate=gate,
            alternate_match=alternate_match,
            elected_stat_vars=elected_stat_vars,
            entity_terms=entity_terms or [],
            ambiguous_entities=ambiguous_entities or [],
            cause=cause,
            unbound_qualifiers=qualifier_terms or [],
            unit_alternatives=unit_alternatives,
            unit_alternatives_truncated=(
                unit_alternatives_truncated if unit_alternatives is not None else None
            ),
            data_query=data_query,
        ),
        rendered_sentence=rendered,
        sentence_plan=plan,
        restatement=restatement,
        restatement_reason=restatement_reason_for(restatement, reason=restatement_reason),
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
    timing_ms: Timing | None = None,
    restatement: ParsedPart | None = None,
    restatement_reason: PartRestatementReason | None = None,
    operations: Sequence[Operation] = (),
    include_skills: bool = False,
    skills_index: SkillsIndex | None = None,
    cause: Cause | None = None,
) -> UnsupportedPart:
    """Wrap an unsupported payload in a part.

    A requested skills match adds ``skills_not_matchable`` because this part has no spec.
    The supplied skills index is unused. The restatement reason is normalised against the
    part's restatement. ``cause`` is populated for ``recipe_not_declared``.
    """
    part_warnings = list(warnings)
    if include_skills:
        part_warnings.append(
            Warning(code=SKILLS_NOT_MATCHABLE, severity="warn", message=_skills_not_matchable_msg())
        )

    return UnsupportedPart(
        status="unsupported",
        variable_text=variable_text,
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
        restatement_reason=restatement_reason_for(restatement, reason=restatement_reason),
        operations=list(operations),
    )


def _with_final_near_neighbor_message(
    warnings: list[Warning],
    clamped: list[Spec],
    near_neighbor_spec_ids: frozenset[str],
) -> list[Warning]:
    """Use final candidate labels in the near-neighbour warning.

    The bind-time warning uses provisional labels. Rebuild it after clamping. With fewer
    than two named specs surviving, drop it, or keep it with
    `NEAR_NEIGHBOR_UNRESOLVED_MESSAGE` when the part states an unevaluated alternate, whose
    unread data leaves the tie standing. Other warning codes pass through unchanged.
    """
    near_neighbor_labels = [
        spec.resolved.shape.label for spec in clamped if spec.spec_id in near_neighbor_spec_ids
    ]
    unevaluated = any(warning.code == ALTERNATES_UNEVALUATED for warning in warnings)
    rebuilt: list[Warning] = []
    for warning in warnings:
        if warning.code != NEAR_NEIGHBOR_AMBIGUOUS:
            rebuilt.append(warning)
            continue
        if len(near_neighbor_labels) < 2:
            if unevaluated:
                rebuilt.append(
                    warning.model_copy(update={"message": NEAR_NEIGHBOR_UNRESOLVED_MESSAGE})
                )
            continue
        rebuilt.append(
            warning.model_copy(
                update={
                    "message": (
                        f"{near_neighbor_labels[0]!r} and {near_neighbor_labels[1]!r} scored "
                        "closely enough to be ambiguous; surfacing both as candidates."
                    )
                }
            )
        )
    return rebuilt


def build_candidates_part(
    specs: list[Spec],
    *,
    variable_text: str | None,
    warnings: list[Warning],
    timing_ms: Timing | None = None,
    max_candidates: int | None = None,
    include_sentence: bool = False,
    include_observations: bool = False,
    bind_ranked: bool = False,
    budget_truncated: bool = False,
    candidate_scores: Mapping[str, float | None] | None = None,
    near_neighbor_spec_ids: frozenset[str] = frozenset(),
    ordering: Ordering | None = None,
    restatement: ParsedPart | None = None,
    restatement_reason: PartRestatementReason | None = None,
    operations: Sequence[Operation] = (),
    include_skills: bool = False,
    skills_index: SkillsIndex | None = None,
    departure_facts: Sequence[DepartureFact] = (),
    request_words: bool = True,
) -> CandidatesPart:
    """Build a candidates part from competing specs.

    The default ``broadest_first`` order sorts by descending member count, then ``spec_id``.
    ``bind_ranked`` preserves the elected-first order from the near-neighbour downgrade path,
    with its near-winning reading next. Bind's preference outranks retrieval score. An
    Explicit ``ordering`` sets the wire label and preserves posted order when the source
    ranking is unavailable.

    Loose readings follow stronger ones, preserving each group's order. A reading is loose
    when its score falls below ``settings().weak_score_threshold`` or its id is in
    ``near_neighbor_spec_ids``. A scoreless candidate is demoted only when its id is flagged.
    The cap defaults to the input length and applies after this partition. ``budget_truncated``
    records upstream omissions and combines with the local cap for ``CandidateSet.truncated``.
    The summary omits a possible-total count when upstream omissions make it unknown.

    Sentence rendering preserves differing how-axis values, states shared departures once, and marks
    the remaining distinctions. ``include_observations`` adds a warning because no candidate
    is selected for observation retrieval. Requested skills attach to each clamped candidate
    when the index snapshot is available. ``departure_facts`` are the raised facts that
    hold for every reading the part shows; no single spec answers, so none is read off one.
    Their text names roles in the flow family every clamped reading shares, and a dataset
    the readings do not come from names their one source, or none when they come from
    several. A part warned `elected_shape_no_data` states one `closest_match` departure
    (`_with_readings_substituted`).
    ``request_words`` is false on a door that holds no words of the request (a resubmit).
    """
    cap = max_candidates if max_candidates is not None else len(specs)

    if ordering is not None or bind_ranked:
        # Preserve the supplied ranking. Posted candidates lack scores for recomputing it.
        sorted_specs = list(specs)
    else:
        sorted_specs = sorted(specs, key=lambda s: (-s.resolved.shape.member_count, s.spec_id))

    def is_loose(spec: Spec) -> bool:
        score = candidate_scores.get(spec.spec_id) if candidate_scores is not None else None
        # Topic-expanded candidates without a score lack a weak-match signal.
        return spec.spec_id in near_neighbor_spec_ids or (
            score is not None and score < settings().weak_score_threshold
        )

    # The cap follows the stable partition, so stronger candidates fill it first.
    ranked_specs = [spec for spec in sorted_specs if not is_loose(spec)]
    ranked_specs.extend(spec for spec in sorted_specs if is_loose(spec))
    clamped = ranked_specs[:cap]
    resolved_ordering: Ordering = (
        ordering if ordering is not None else ("intent_rank" if bind_ranked else "broadest_first")
    )
    # Report either upstream budget omissions or this function's own cap.
    truncated = budget_truncated or len(ranked_specs) > cap

    keep_how_properties = sibling_how_properties(clamped)
    if include_sentence:
        clamped = [with_sentence(spec, keep_how_properties=keep_how_properties) for spec in clamped]
        # Marks the tokens that set each reading apart (`SentenceToken.distinguishes`).
        candidate_differences(clamped, axis_declarations=declared_axis_names())

    if include_skills and skills_index is not None:
        # Skill matching is independent of sentence rendering.
        clamped = [with_skills(spec, index=skills_index) for spec in clamped]

    departures = part_departures(
        _with_readings_substituted(
            _with_source_shown(
                part_departure_facts(None, raised=departure_facts, operations=operations),
                shown=clamped,
            ),
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
    plan: list[SentenceToken] | None = None
    rendered: str | None = None
    if include_sentence:
        plan = candidates_sentence_plan(
            clamped,
            departures=departures,
            truncated=truncated,
            # Show the pre-clamp total only when no upstream budget hid candidates.
            total=len(ranked_specs) if not budget_truncated else None,
        )
        rendered = sentence_from_plan(plan)

    part_warnings = _with_final_near_neighbor_message(warnings, clamped, near_neighbor_spec_ids)
    if include_observations:
        part_warnings.append(
            Warning(code=OBSERVATIONS_OMITTED, severity="warn", message=_observations_omitted_msg())
        )
    # Every candidate repeats its members, so a family candidate lists fewer than a
    # definite answer. The sentence above read them all.
    family_axes = get_measure_index().axes
    clamped = [
        capping_stat_var_items(spec, cap=CANDIDATE_STAT_VARS_MAX)
        if identity_names_measure_family(spec.definition.identity, family_axes)
        else spec
        for spec in clamped
    ]

    return CandidatesPart(
        status="candidates",
        variable_text=variable_text,
        warnings=part_warnings,
        timing_ms=timing_ms,
        candidates=CandidateSet(
            ordering=resolved_ordering,
            max_candidates=cap,
            specs=clamped,
            truncated=truncated,
        ),
        rendered_sentence=rendered,
        sentence_plan=plan,
        restatement=restatement,
        restatement_reason=restatement_reason_for(restatement, reason=restatement_reason),
        operations=list(operations),
        departures=departures,
    )
