"""Pure assembly and grouping for grounded Search regions.

Every input region yields a Part in query order. Definite regions share a spec
only after a successful same-shape collapse. Folded parts receive
``BINDING_FOLDED``. Non-definite and refused regions keep their own payloads.
``combine_regions`` returns the post-collapse regions for task reconciliation.
This module makes no graph or LLM calls.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

from pydantic import ValidationError

from origin_catalogue import current_catalogue
from search.engine.assemble import (
    build_candidates_part,
    build_definite_part,
    build_no_data_part,
    build_unsupported_part,
    skills_index_incomplete_warning,
    skills_index_unavailable_warning,
)
from search.engine.departures import surviving_facts
from search.engine.entity_checks import part_entities
from search.engine.envelope import make_diagnostics, make_query_echo, make_response, now_ms
from search.engine.errors import EngineInputError
from search.engine.region.source_choice import scope_records
from search.engine.role_text import (
    parsed_directions_contradicted,
    role_axis_for_text,
    role_text_for_axis,
)
from search.engine.same_shape import (
    FacetOptionsReader,
    ScopeReader,
    collapse_same_shape,
    fold_key,
)
from search.engine.source_scope import compile_part_sources
from search.engine.unit.bind import unit_text_naming
from search.engine.unit.conversion import conversion_unavailable_warning
from search.models import (
    AmbiguousEntityCandidate,
    AmbiguousEntityGroup,
    AmbiguousSourceCandidate,
    AmbiguousSourceGroup,
    EntityRefs,
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
from search.render.labels import join_labels
from search.render.no_data import ScopeSources, UnlistedPopulation, untold_place_name
from skills_index import SkillsIndex

if TYPE_CHECKING:
    from search.engine.region import RegionResult
    from search.engine.region.composite_answer import ComponentReads
    from search.engine.region.result import AmbiguousEntityCandidate as EngineCandidate
    from search.engine.region.result import AmbiguousSource
    from search.models import GraphRef, Part, QueryEcho

# ---------------------------------------------------------------------------
# Warning code and pinned severity/message
# ---------------------------------------------------------------------------

BINDING_FOLDED = "binding_folded"
VARIABLES_CLAMPED = "variables_clamped"

# The steps every leg's own timing_by_step seeds from the shared, once-per-request stages
# ahead of the fan-out, extraction and the exclusion read (see fanout.py's base_timing). Each
# is reported once, on the envelope's own timing_ms.by_step.
_SHARED_TIMING_STEPS = frozenset({"extract", "exclusion_read"})

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
    no region records a shared step.
    """
    for region in regions:
        shared = {
            step: ms for step, ms in region.timing_by_step.items() if step in _SHARED_TIMING_STEPS
        }
        if shared:
            return shared
    return {}


def _entity_refs(region: RegionResult, restatement: ParsedPart | None) -> list[EntityRefs]:
    """The binding of each row of the part's `restatement`, one per row in row order
    (`Part.entity_refs`), looked up by the row's identity (`ParsedEntity.identity`) in
    `RegionResult.entity_refs`: a self-flow's two rows state the same refs, and "Congo" to
    "Congo" states each end's. A row the region binds nothing for states []."""
    if restatement is None:
        return []
    return [
        EntityRefs(text=entity.text, refs=list(region.entity_refs.get(entity.identity, ())))
        for entity in restatement.entities or ()
    ]


def _asked_words(region: RegionResult) -> dict[str, str]:
    """The words resolution read for each ambiguous entity name of `region`
    (``RegionResult.entity_surfaces``), else the name, which its refusal quotes whatever
    the refusal's reason and its ambiguous rows post (`_asked_row`): "Congo" for the name
    "Democratic Republic of the Congo". The result is keyed by name, so two names written
    alike stay two choices."""
    return {e.text: region.entity_surfaces.get(e.text, e.text) for e in region.ambiguous_entities}


def _asked_row(entity: ParsedEntity, asked: Mapping[str, str]) -> ParsedEntity:
    """`entity`, a row of a region's parsed form, posted as the parsed door reads it back
    to the same outcome: an ambiguous name's row posts its asked words (`_asked_words`) as
    its text and the name as its identity (`ParsedEntity.name`), so "aid from Congo to
    Congo" posts two "Congo" rows under two names and asks two choices again. Every other
    row posts as it is."""
    identity = entity.identity
    if identity not in asked:
        return entity
    text = asked[identity]
    return entity.model_copy(update={"text": text, "name": None if text == identity else identity})


def _asked_restatement(region: RegionResult, asked: Mapping[str, str]) -> ParsedPart | None:
    """The region's restatement with each ambiguous row as the parsed door reads it back
    (`_asked_row`, from the region's `_asked_words`), so posting it back asks the same
    choice."""
    if region.restatement is None or not asked:
        return region.restatement
    entities = [_asked_row(entity, asked) for entity in region.restatement.entities or ()]
    return region.restatement.model_copy(update={"entities": entities})


def _no_data_entity_terms(
    region: RegionResult, asked: Mapping[str, str]
) -> tuple[list[str] | None, list[str] | None]:
    """Return entity terms and candidates for no-data reasons that carry them.

    Ambiguous entities supply terms and ordered candidate names. Role-not-admitted
    entities supply terms only, in the request's own words (``RegionResult.entity_words``).
    Unresolved entities use ``unresolved_entity_names`` in the request's own words: an
    ambiguous one in the words its resolution read (`asked`, the region's `_asked_words`),
    any other through ``RegionResult.words_of``. So do the names whose direction a
    ``role_not_resolved`` refusal could not tell and the whole populations a
    ``role_incompatible`` refusal found directed to the other side, each in the request's
    own words for it (``RegionResult.entity_words``). Other reasons or empty values return
    ``None``.
    """
    if region.no_data_reason == "entity_ambiguous":
        entity_terms = [asked[e.text] for e in region.ambiguous_entities] or None
        candidates = list(
            dict.fromkeys(c.name for e in region.ambiguous_entities for c in e.candidates)
        )
        return entity_terms, candidates or None
    if region.no_data_reason == "entity_role_not_admitted":
        entity_terms = [region.words_of(e.text) for e in region.role_not_admitted_entities]
        return entity_terms or None, None
    if region.no_data_reason == "entity_not_resolved":
        terms = [
            asked.get(name) or region.words_of(name) for name in region.unresolved_entity_names
        ]
        return terms or None, None
    if region.no_data_reason in ("role_not_resolved", "role_incompatible"):
        return _untold_terms(region) or None, None
    return None, None


def _unlisted_populations(region: RegionResult) -> dict[str, UnlistedPopulation]:
    """Each population an ``entity_role_not_admitted`` region refuses because no shape lists
    it exactly (`RoleNotAdmittedEntity.unlisted`), by the words the refusal quotes for it
    (`_no_data_entity_terms`)."""
    return {
        region.words_of(entity.text): entity.unlisted
        for entity in region.role_not_admitted_entities
        if entity.unlisted is not None
    }


def _untold_terms(region: RegionResult) -> list[str]:
    """Every request text a ``role_not_resolved`` or ``role_incompatible`` refusal names
    (``RegionResult.unresolved_entity_names``): each place set whose direction the request
    does not tell, or each whole population directed to the other side of the flow. In query
    order and in the request's own words (``RegionResult.entity_words``), each once per
    place (``RegionResult.untold_groups``), so two places written alike keep a term each:
    the refusal's entity terms and every role option's mentions."""
    place_of = {name: i for i, group in enumerate(region.untold_groups) for name in group}
    terms = {
        (place_of.get(name), region.words_of(name)): region.words_of(name)
        for name in region.unresolved_entity_names
    }
    return list(terms.values())


def _untold_place_terms(region: RegionResult) -> list[list[str]] | None:
    """The terms of a ``role_not_resolved`` refusal grouped by the place each names
    (`_untold_place_words`), so the sentence counts places. ``None`` for every other
    reason."""
    if region.no_data_reason != "role_not_resolved":
        return None
    return list(_untold_place_words(region).values()) or None


def _untold_place_words(region: RegionResult) -> dict[tuple[str, ...], list[str]]:
    """The words naming each untold place (``RegionResult.untold_groups``), by its group of
    request texts: those texts in the request's own words as `_untold_terms` gives them,
    each once. A place whose words name another untold place too ends with its graph label
    (``RegionResult.untold_labels``), so "Congo" and "Congo" read as two places."""
    named = [
        list(dict.fromkeys(region.words_of(name) for name in group))
        for group in region.untold_groups
    ]
    return {
        group: [*terms, label] if label and named.count(terms) > 1 else terms
        for group, terms, label in zip(
            region.untold_groups, named, region.untold_labels, strict=True
        )
    }


def _ambiguous_entity_groups(
    region: RegionResult, asked: Mapping[str, str]
) -> list[AmbiguousEntityGroup]:
    """Convert region ambiguity data to the wire shape, for an ``entity_ambiguous`` region
    only. A region refused for another reason can still name ambiguous entities
    (`phase_intake._phase_check_entities`), and the wire field is empty for it.

    Engine candidates use ``name``. Wire candidates use ``label``. Each wording reads in the
    request's own words (`asked`, the region's `_asked_words`).
    """
    if region.no_data_reason != "entity_ambiguous":
        return []
    return [
        AmbiguousEntityGroup(
            text=asked[entity.text],
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


def _option(
    mentions: Sequence[str], label: str, part: ParsedPart, *, entity_choice: int | None = None
) -> NoDataOption | None:
    """One option posting `part`, settling the ambiguous entity group at `entity_choice`
    when it settles one, or `None` when the parsed door would refuse it: `part`
    breaks the `ParsedPart` bounds, states a direction its own wording contradicts, posts
    entities the door's checks refuse (`entity_checks.part_entities`), or names sources that leave
    it none (`source_scope.compile_part_sources`, which reads the bound catalogue's cached
    dictionary)."""
    try:
        posted = ParsedPart.model_validate(part.model_dump())
    except ValidationError:
        return None
    if parsed_directions_contradicted(posted):
        return None
    try:
        part_entities(posted, pointer_prefix="")
        if posted.sources:
            compile_part_sources(posted, catalogue=current_catalogue(), pointer_prefix="")
    except EngineInputError:
        return None
    return NoDataOption(
        mentions=list(mentions),
        label=label,
        data_query=ParsedInput(parts=[posted]),
        entity_choice=entity_choice,
    )


def _entity_options(
    region: RegionResult, part: ParsedPart, *, asked: Mapping[str, str]
) -> list[NoDataOption | None]:
    """One option per ambiguous wording and candidate: the part with every entity that
    resolved to one node pinned, and every row with the wording pinned to the candidate.
    A pinned row posts its words and the dcid, which is its identity, with no name. A row
    asked for its members takes a group's own node by the group's registered name, with no
    dcid, since a pin cannot ask for a group's members; a group node with no registered
    name that resolves to it alone has no option. Every other ambiguous row posts as the
    parsed door reads it back to the same outcome (`_asked_row`, from `asked`, the region's
    `_asked_words`), so it asks its choice again and "aid from Congo to Congo" settles one
    end per option. The option's mentions quote the asked words. Each option names its
    group's index in `region.ambiguous_entities` (`NoDataOption.entity_choice`), since two
    groups can share one wording."""

    def pinned(entity: ParsedEntity) -> ParsedEntity:
        dcid = region.resolved_entity_dcids.get(entity.identity)
        if dcid is None:
            return entity
        return entity.model_copy(update={"name": None, "dcid": dcid})

    def chosen(entity: ParsedEntity, candidate: EngineCandidate) -> ParsedEntity | None:
        if entity.members and candidate.group_node:
            if candidate.group_name is None:
                return None
            return entity.model_copy(
                update={"text": candidate.group_name, "name": None, "dcid": None}
            )
        return entity.model_copy(update={"name": None, "dcid": candidate.dcid})

    base = [_asked_row(pinned(entity), asked) for entity in part.entities or ()]
    options: list[NoDataOption | None] = []
    for i, group in enumerate(region.ambiguous_entities):
        for candidate in group.candidates:
            entities = [
                chosen(entity, candidate) if entity.identity == group.text else entity
                for entity in base
            ]
            if None in entities:
                continue
            options.append(
                _option(
                    [asked[group.text]],
                    candidate.name,
                    part.model_copy(update={"entities": entities}),
                    entity_choice=i,
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
    the reading repeats is dropped. Its mentions are the refusal's own entity terms
    (`_untold_terms`), the same for every reading. The label reads the option's own
    directions (`_direction_label`)."""
    mentions = _untold_terms(region)
    options: list[NoDataOption | None] = []
    for reading in region.role_assignments:
        rows = [
            entity.model_copy(update={"role_text": role_text_for_axis(reading[entity.identity])})
            if entity.identity in reading
            else entity
            for entity in part.entities or ()
        ]
        entities = [row for i, row in enumerate(rows) if row not in rows[:i]]
        label = _direction_label(entities, region)
        options.append(_option(mentions, label, part.model_copy(update={"entities": entities})))
    return options


def _direction_label(entities: Sequence[ParsedEntity], region: RegionResult) -> str:
    """The directions `entities` state, in words. Origins read "to" destinations, and one
    axis alone reads "From" or "To" with its names. Each untold place set
    (``RegionResult.untold_groups``) is named once in its words (`_untold_place_words`), as
    the refusal's message names it (`untold_place_name`), as in `Kenya ("KEN") to Norway`."""
    place_words = _untold_place_words(region)
    group_of = {name: group for group in region.untold_groups for name in group}
    by_axis: dict[RoleAxis, list[tuple[str, ...]]] = {"origin": [], "destination": []}
    for entity in entities:
        axis = role_axis_for_text(entity.role_text)
        group = group_of.get(entity.identity, (entity.identity,))
        if axis is not None and group not in by_axis[axis]:
            by_axis[axis].append(group)
    origins, destinations = (
        _axis_label(
            [
                place_words.get(group) or [region.words_of(text) for text in group]
                for group in by_axis[axis]
            ]
        )
        for axis in ("origin", "destination")
    )
    if origins and destinations:
        return f"{origins} to {destinations}"
    return f"From {origins}" if origins else f"To {destinations}"


def _axis_label(places: Sequence[Sequence[str]]) -> str:
    """The places `places` holds, each by its words."""
    names = (untold_place_name(texts, quote_first=False) for texts in places)
    return join_labels([name for name in names if name])


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
    "source_ambiguous": _source_options,
    "source_code_ambiguous": _source_code_options,
    "role_not_resolved": _role_options,
    "unit_not_resolved": _unit_options,
}


def _options(region: RegionResult, *, asked: Mapping[str, str]) -> list[NoDataOption]:
    """The ready requests of a refusal that asks for a choice, each a one-part parsed input
    built from the region's parsed form; an entity choice reads the region's `_asked_words`
    as `asked`. Empty for every other reason, when the region has no parsed form (its
    inputs exceed the `ParsedPart` bounds), and for an option the parsed door would refuse
    (`_option`)."""
    reason, part = region.no_data_reason, region.parsed_form
    if reason is None or part is None:
        return []
    if reason == "entity_ambiguous":
        built = _entity_options(region, part, asked=asked)
    elif (build := _OPTION_BUILDERS.get(reason)) is not None:
        built = build(region, part)
    else:
        return []
    return [option for option in built if option is not None]


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

    Variable text, restatement, operations, and the declared defaults a definite part
    assumes belong to the original region, even when a merged representative supplies its
    spec. Shared timing appears on the envelope, so part timing excludes it. Folded parts
    carry only ``binding_folded``. The representative's warnings appear on its first part.

    A raised departure fact reaches the part only while the part's or the envelope's warnings,
    or the request-wide qualifier warnings its own region leaves unapplied
    (`RegionResult.unapplied_qualifier_warnings`), hold the warning it was raised beside. On
    a candidates part only warnings stamped with no reading's `spec_id` count, since the
    part's departures speak for every reading it shows.
    """
    own_timing_by_step = {
        step: ms for step, ms in region.timing_by_step.items() if step not in _SHARED_TIMING_STEPS
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
            entity_refs=_entity_refs(region, region.restatement),
            operations=region.operations,
            include_skills=include_skills,
            skills_index=skills_index,
            departure_facts=(
                *surviving_facts(
                    raised, [*warnings, *region.unapplied_qualifier_warnings, *envelope_warnings]
                ),
                *(
                    ()
                    if folded
                    else (
                        *representative.inferred_facts,
                        *representative.unpublished_whole_facts.get(
                            representative.spec.spec_id, ()
                        ),
                    )
                ),
            ),
            entity_words=region.entity_words,
            assumption_echo=region.assumption_echo,
            name_corrections=region.name_corrections,
            asked_axes=region.every_member_axes,
            own_spec=None if representative is region else region.spec,
            members_in_words=region.members_in_words,
            scope_choices=representative.scope_choices,
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
            entity_refs=_entity_refs(region, region.restatement),
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
            reading_facts=region.unpublished_whole_facts,
            request_words=region.entity_words is not None,
            name_corrections=region.name_corrections,
            asked_axes=region.every_member_axes,
            assumption_echoes=region.reading_assumption_echoes,
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
            entity_refs=_entity_refs(region, region.restatement),
            operations=region.operations,
            include_skills=include_skills,
            skills_index=skills_index,
            cause=region.unsupported.cause,
        )
    assert region.no_data_reason, (
        "a no_data region must carry a real reason code; a falsy no_data_reason "
        "here is an assembly bug, not a case to default around"
    )
    asked = _asked_words(region)
    entity_terms, ambiguous_candidates = _no_data_entity_terms(region, asked)
    if region.no_data_reason in ("source_ambiguous", "source_code_ambiguous"):
        ambiguous_candidates = [
            c.label for group in region.ambiguous_sources for c in group.candidates
        ] or None
    qualifier_terms, unbound_sources, scope = _unbound_terms(region)
    restatement = _asked_restatement(region, asked)
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
        unlisted=_unlisted_populations(region),
        entity_groups=_untold_place_terms(region),
        role_axis=region.no_data_role_axis,
        bound_role_dcids=region.no_data_bound_role_dcids,
        ambiguous_candidates=ambiguous_candidates,
        ambiguous_entities=_ambiguous_entity_groups(region, asked),
        ambiguous_roles=[entity.role for entity in region.ambiguous_entities],
        options=_options(region, asked=asked),
        source_terms=list(region.source_terms) or None,
        ambiguous_sources=_ambiguous_source_groups(region),
        qualifier_terms=qualifier_terms,
        unbound_sources=unbound_sources,
        scope=scope,
        unit_alternatives=list(region.unit_alternatives) or None,
        denominator=region.denominator,
        restatement=restatement,
        entity_refs=_entity_refs(region, restatement),
        operations=region.operations,
        include_skills=include_skills,
        skills_index=skills_index,
        cause=region.cause,
        window_gap_fact=region.window_gap_fact,
        departure_facts=surviving_facts(
            raised,
            [*region.warnings, *region.unapplied_qualifier_warnings, *envelope_warnings],
        ),
        reading_facts=region.unpublished_whole_facts,
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
    # Stamped once the part is built, so the total covers its assembly.
    diag = make_diagnostics(
        engine_build,
        envelope_warnings,
        _shared_timing_by_step([region]),
        now_ms() - start_ms,
        llm_usage=llm_usage,
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
    scope_for: ScopeReader | None = None,
    preview_reads: ComponentReads | None = None,
) -> tuple[Resolution, list[RegionResult]]:
    """Combine regions into one response, with one part per input in query order.

    Same-shape definite regions merge only when collapse succeeds. Merged
    non-first parts share the representative spec and get ``binding_folded``.
    Refused groups and other regions keep their own payloads.

    Returns the response and post-collapse regions. Untouched entries retain
    object identity. Merged entries are new with ``slot_options_task`` and
    ``place_options_task`` cleared. Identity lets the caller cancel and relaunch enrichment.
    Its only graph reads are the injected ``set_ref_for`` and ``facet_options_for``
    readers a same-shape merge calls, and ``scope_for`` builds a merged region's scope,
    which every part it folds states. A merge of composite answers reads its components over
    ``preview_reads``, the request's component reads, and composite answers stay apart
    without them. It performs no task operations.

    ``llm_usage`` is summed by the caller. A supplied ``echo`` is used
    verbatim. Otherwise a raw-text echo is built from ``query``. Skills-index
    warnings are added once on the envelope.
    """
    collapsed, residual_texts = collapse_same_shape(
        [r for r in regions if r.status == "definite"],
        set_ref_for=set_ref_for,
        facet_options_for=facet_options_for,
        scope_for=scope_for,
        preview_reads=preview_reads,
    )
    return combine_collapsed(
        regions,
        collapsed,
        residual_texts,
        extra_warnings=extra_warnings,
        start_ms=start_ms,
        engine_build=engine_build,
        max_candidates=max_candidates,
        include_observations=include_observations,
        include_skills=include_skills,
        skills_index=skills_index,
        llm_usage=llm_usage,
        echo=echo,
        query=query,
    )


def combine_collapsed(
    regions: list[RegionResult],
    collapsed: list[RegionResult],
    residual_texts: list[tuple[str | None, int]],
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
) -> tuple[Resolution, list[RegionResult]]:
    """Combine `regions` into one response as `combine_regions` does, over `collapsed` and
    `residual_texts`, what `collapse_same_shape` returned for their definite regions. A
    caller that settles the merged regions between the two steps calls them apart
    (`entry.multi._stream_combined_regions`)."""
    definite = [r for r in regions if r.status == "definite"]
    others = [r for r in regions if r.status != "definite"]

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
    # Stamped once the parts are built, so the total covers their assembly.
    diag = make_diagnostics(
        engine_build,
        envelope_warnings,
        _shared_timing_by_step(regions),
        now_ms() - start_ms,
        llm_usage=llm_usage,
    )
    response = make_response(
        echo, diag, parts, include_skills=include_skills, skills_index=skills_index
    )
    return response, effective
