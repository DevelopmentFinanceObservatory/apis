"""The typed heading and byline a resolved Spec states, built before any text.

`heading_for` reads a Spec and `heading_from_parts` the bind-time parts; both return a
`HeadingPlan`. Every phrase is chosen here: the possessive lead, the measure noun and the
axes it states, the how values, the roles, the subject, the period, and the byline's unit,
source and bound measurement method. `render.tokens` lays a `HeadingPlan` out as sentence tokens, so casing,
spacing and punctuation live there alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, NamedTuple, Protocol

from code_index.flow_family import flow_family
from code_index.labels import PRICE_BASIS_AXIS
from code_index.shared import get_index
from dfo_platform.roles import FlowFamily, RoleNoun, RoleWording, role_wording
from search.heading.pins import identity_pinned_properties
from search.models import (
    UNIT_REF,
    BindingSet,
    BindingValue,
    Coverage,
    CoverageBreadth,
    CoverageExact,
    DateRange,
    Entity,
    EntityGroup,
    EntityRoleDirectional,
    GraphRef,
    IdentityBindingSet,
    IdentityBindingValue,
    IdentityCurrencyRole,
    IdentityFacetValueAbsent,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    OpenConstraintAxisFact,
    ResolvedEntity,
    ResolvedFacet,
    ResolvedScope,
    RoleAxis,
    RoleAxisFact,
    SentenceFacts,
    Slot,
    SlotValue,
    Spec,
    TimeWindow,
)
from unit_words.currency_nouns import currency_noun
from unit_words.price_basis import PriceBasis, price_basis, price_basis_phrase
from unit_words.unit_class import classify_unit

from .how import HowValue, _shared_constraint_values, how_values
from .labels import (
    _binding_labels,
    _binding_values,
    _entity_labels,
    _entity_name,
    _humanize,
    _humanize_ref,
    _humanize_source_label,
    _join_labels,
    axis_ref,
    group_text,
)
from .what import (
    _authored_measure_noun,
    _fallback_what,
    _flow_short_names,
    _lower_first_word,
    _measure_extras,
    _names_currency,
    _unit_heading_phrase,
    _with_short_names,
    facet_currency_units,
)
from .window import Period, served_period

__all__ = [
    "LATEST_YEAR_LABEL",
    "Byline",
    "Heading",
    "HeadingPlan",
    "HowValue",
    "MeasureNoun",
    "NounKind",
    "Period",
    "RolePhrase",
    "ShapeLike",
    "SourcePhrase",
    "declared_axis_names",
    "heading_for",
    "heading_from_parts",
    "sibling_how_properties",
    "value_signature",
]


class ShapeLike(Protocol):
    """Shape fields needed to name the measure before binding completes."""

    label: str
    measured_property: GraphRef | None
    population_type: GraphRef | None
    stat_type: GraphRef | None
    measurement_qualifier: GraphRef | None
    measurement_denominator: GraphRef | None


LATEST_YEAR_LABEL = "Latest year with data"
"""The byline's opening item for a request that asked for the latest year."""


NounKind = Literal["bound", "label", "authored", "stat_var_name", "graph_name"]
"""Where a measure noun comes from: a bound what value's label, the engine's measure
label (`ResolvedSpec.measure_label`), an authored IDS or WEO noun (`measure_noun`), one
StatVar's name, or a graph name from the shape (property, population type, label)."""


@dataclass(frozen=True, slots=True)
class MeasureNoun:
    """The noun naming the measure and the how axes it states.

    ``measure_axes`` are the measure-partition axes a label or authored noun states and
    the flow type whose short name it carries ("ODA"). ``named_axes`` are the how axes
    whose shared value the noun's text names, as its code-index entry declares
    (`MeasurePartitionLabel.named_values`: "imputed multilateral ODA" names
    `DAC2aAidType106`). Neither holds anything for a graph name, whose values
    `how_values` reads in its text. ``states`` are the refs the noun's token states
    besides its own slot (`_noun_states`).
    """

    text: str
    kind: NounKind
    slot: Slot | None
    measure_axes: tuple[str, ...] = ()
    named_axes: tuple[str, ...] = ()
    states: tuple[str, ...] = ()

    @property
    def stated(self) -> bool:
        """Whether a measure label or authored noun states the measure."""
        return self.kind in ("label", "authored")

    @property
    def consumed_axes(self) -> frozenset[str]:
        """Every how axis the noun states, which no how value repeats."""
        return frozenset((*self.measure_axes, *self.named_axes))

    @property
    def inline_text(self) -> str:
        """The noun as it reads inside a sentence, its first word lowered when plain
        Title Case ("external debt stocks")."""
        return _lower_first_word(self.text)


@dataclass(frozen=True, slots=True)
class RolePhrase:
    """A flow role's entities or the subject place, as the heading names them.

    ``axis`` is the flow role, None for the subject place. ``preposition`` joins the
    phrase after the noun ("to", "from", "in"). None reads it as body text ("worldwide",
    "across 78 countries"). ``entity_dcids`` are the named entities in display order,
    empty for an open role read as a count. ``slot`` backs the phrase, the where slot of
    the subject or of the destination. ``possessable`` marks one named entity outside a group, which
    can lead the heading in possessive form.
    """

    axis: RoleAxis | None
    preposition: str | None
    text: str
    entity_dcids: tuple[str, ...]
    slot: Slot | None
    possessable: bool


@dataclass(frozen=True, slots=True)
class SourcePhrase:
    """The byline's source: a bound source constraint or the answer's resolved
    provenance. ``text`` is empty for an unbound source slot with no provenance, which
    keeps its place."""

    text: str
    slot: Slot | None


@dataclass(frozen=True, slots=True)
class Heading:
    """What the answer is, in possessive order: lead, modifiers, noun, extras, unit,
    roles, how values, subject, period.

    ``lead`` is the possessive actor ("Germany's"). ``modifiers`` are how values before
    the noun, ``how`` those after the roles, silent ones included. ``extras`` qualify a
    graph name (stat type, measurement qualifier). ``unit`` is the unit phrase that
    changes the magnitude ("per capita", "in percent"), and ``unit_states`` the refs its
    token states: `UNIT_REF` when the phrase names the unit (`UnitPhrase.names_unit`).
    ``roles`` follow the noun, destination first.
    ``subject`` is the where slot's phrase when no destination names the place.
    ``when_slot`` backs the period. ``family`` is the flow family the members share.
    """

    lead: RolePhrase | None
    modifiers: tuple[HowValue, ...]
    noun: MeasureNoun
    extras: tuple[str, ...]
    unit: str | None
    unit_states: tuple[str, ...]
    roles: tuple[RolePhrase, ...]
    how: tuple[HowValue, ...]
    subject: RolePhrase | None
    period: Period | None
    when_slot: Slot | None
    family: FlowFamily | None


@dataclass(frozen=True, slots=True)
class Byline:
    """The byline after the heading: the latest-year label under ``latest``, the currency
    with its price basis (``unit_text``, "current US dollars", "constant 2024 euros"),
    the source, and the measurement method the identity binds (``method``, None when it
    binds none). ``unit_states`` are the refs the unit token states besides `UNIT_REF`:
    the price-basis axis when ``unit_text`` names one."""

    latest: bool
    unit_text: str | None
    source: SourcePhrase | None
    unit_states: tuple[str, ...] = ()
    method: str | None = None


@dataclass(frozen=True, slots=True)
class HeadingPlan:
    """A Spec's heading and byline."""

    heading: Heading
    byline: Byline


class _NormalizedEntity(NamedTuple):
    """Common dcid, label, and role view of engine and response entity models."""

    dcid: str
    label: str | None
    axis: RoleAxis | None  # None for the observationAbout/subject role


def _normalize_entities(entities: list[Entity] | list[ResolvedEntity]) -> list[_NormalizedEntity]:
    """Normalise each engine or response entity to the heading's common view."""
    result: list[_NormalizedEntity] = []
    for e in entities:
        if isinstance(e, Entity):
            axis = e.role.axis if isinstance(e.role, EntityRoleDirectional) else None
            result.append(_NormalizedEntity(dcid=e.ref.ref, label=e.ref.label, axis=axis))
        else:
            axis = e.role if e.role in ("origin", "destination") else None
            result.append(_NormalizedEntity(dcid=e.ref, label=e.label, axis=axis))
    return result


def heading_for(
    spec: Spec,
    *,
    axis_declarations: Mapping[str, str] | None = None,
    keep_how_properties: frozenset[str] = frozenset(),
) -> HeadingPlan:
    """The heading and byline a resolved Spec states (`heading_from_parts`).

    Resolved StatVars, sources, coverage, scope, facets, open-axis facts, the measure
    label, entity groups and role-axis facts fill the plan from the answer.
    ``axis_declarations`` supplies how-property names and ordering, the code index's
    (`declared_axis_names`) when None. ``keep_how_properties`` keeps
    sibling-distinguishing constraints visible. The slots the identity's five-tuple pins
    (`identity_pinned_properties`) state no how clause. The byline states the
    measurement method the identity binds (`_bound_method_text`).
    """
    resolved = spec.resolved
    plan = heading_from_parts(
        resolved.slots,
        resolved.entities,
        resolved.shape,
        resolved.coverage.window,
        latest=spec.definition.identity.requested_window.kind == "latest",
        facts=resolved.sentence_facts,
        served=resolved.stat_vars.data_date_range,
        resolved_sources=resolved.trace.resolved_sources,
        coverage=resolved.coverage,
        scope=resolved.scope,
        facets=resolved.facets,
        keep_how_properties=keep_how_properties,
        pinned_properties=identity_pinned_properties(spec.definition.identity),
        open_axis_facts=resolved.open_axis_facts or (),
        axis_declarations=(
            declared_axis_names() if axis_declarations is None else axis_declarations
        ),
        measure_label=resolved.measure_label,
        measure_label_axes=resolved.measure_label_axes,
        measure_label_named_values=resolved.measure_label_named_values,
        unit_request=_unit_request(spec),
        entity_groups=resolved.entity_groups,
        role_axis_facts=resolved.role_axis_facts or (),
    )
    method = _bound_method_text(spec)
    if method is None:
        return plan
    return replace(plan, byline=replace(plan.byline, method=method))


NO_STATED_METHOD = "no stated method"
"""The byline's method text when the identity binds `measurementMethod` to a null ref."""


def _bound_method_text(spec: Spec) -> str | None:
    """The measurement method the identity binds, as the byline states it, or None when
    it binds none. A null ref reads `NO_STATED_METHOD`. A ref, or a set whose served
    series share one method, reads the served facet's label (`ResolvedFacet.label`). A
    set the series serve per series, and a ref the facets do not label, read each bound
    ref's humanised dcid (`_humanize_ref`), joined."""
    binding = (spec.definition.identity.facets or {}).get("measurementMethod")
    if binding is None:
        return None
    if isinstance(binding, IdentityFacetValueAbsent):
        return NO_STATED_METHOD
    if isinstance(binding, IdentityBindingValue):
        refs = [binding.ref]
    elif isinstance(binding, IdentityBindingSet):
        refs = binding.refs
    else:
        return None
    served = spec.resolved.facets.get("measurementMethod")
    if served is not None and served.ref is not None and served.ref in refs:
        return _humanize_ref(served.ref, served.label)
    return _join_labels([_humanize_ref(ref, None) for ref in refs])


def declared_axis_names() -> dict[str, str]:
    """The code index's how-property dcids mapped to their plain names, in declaration
    order. These are the ``axis_declarations`` every Spec's sentence plan is rendered
    with."""
    return {dcid: declaration.plain_name for dcid, declaration in get_index().declarations.items()}


# An unbound slot's signature, distinct from every bound value set.
_UNBOUND_HOW_SIGNATURE: frozenset[str] = frozenset()


def sibling_how_properties(specs: Sequence[Spec]) -> frozenset[str]:
    """The how properties whose bindings differ across sibling Specs, which each sibling's
    heading keeps visible (``keep_how_properties``) so folding cannot make two siblings
    read alike. A bound slot's signature is its values' dcids or literals. An unbound
    slot's is empty and differs from every bound one."""
    signatures: dict[str, set[frozenset[str]]] = {}
    for spec in specs:
        for slot in spec.resolved.slots:
            if slot.key.axis != "how" or slot.key.property is None:
                continue
            binding = slot.binding
            if isinstance(binding, BindingValue):
                signature = frozenset({value_signature(binding.value)})
            elif isinstance(binding, BindingSet):
                signature = frozenset(value_signature(v) for v in binding.values)
            else:
                signature = _UNBOUND_HOW_SIGNATURE
            signatures.setdefault(slot.key.property.ref, set()).add(signature)
    return frozenset(prop for prop, seen in signatures.items() if len(seen) > 1)


def value_signature(value: SlotValue) -> str:
    """A slot value's ref dcid or literal, which display-label changes leave alone."""
    return value.ref.ref if value.ref is not None else (value.literal or "")


def _unit_request(spec: Spec) -> IdentityUnitRequest | None:
    """The identity's unit request, when the request converts the answer."""
    unit_binding = (spec.definition.identity.facets or {}).get("unit")
    return unit_binding.request if isinstance(unit_binding, IdentityFacetValueRequest) else None


def heading_from_parts(
    slots: list[Slot],
    entities: list[Entity] | list[ResolvedEntity],
    shape: ShapeLike,
    window: TimeWindow | None,
    *,
    latest: bool = False,
    facts: SentenceFacts | None = None,
    served: DateRange | None = None,
    resolved_sources: Sequence[GraphRef] | None = None,
    coverage: Coverage | None = None,
    scope: ResolvedScope | None = None,
    facets: Mapping[str, ResolvedFacet] | None = None,
    keep_how_properties: frozenset[str] = frozenset(),
    pinned_properties: frozenset[str],
    open_axis_facts: Sequence[OpenConstraintAxisFact] = (),
    axis_declarations: Mapping[str, str] | None = None,
    measure_label: str | None = None,
    measure_label_axes: Sequence[str] = (),
    measure_label_named_values: Sequence[tuple[str, str]] = (),
    unit_request: IdentityUnitRequest | None = None,
    entity_groups: Mapping[str, EntityGroup] | None = None,
    role_axis_facts: Sequence[RoleAxisFact] = (),
    member_dcids: Sequence[str] = (),
) -> HeadingPlan:
    """Build a heading in possessive order (actor, measure noun, counterpart, period),
    then its byline (latest-year label, unit, source).

    The actor leads in possessive form when it names one entity: the flow family's
    possessor role (`RoleWording.possessor`: the provider for aid, the debtor for debt),
    or the subject place when no flow role applies. "Germany's gross ODA disbursements
    to Kenya", "Japan's GDP per capita in PPP international dollars". Every other role
    follows the noun with its family preposition, destination first ("owed to China",
    "from the 27 EU member states"), and before the how values the heading states ("to
    Nigeria for basic health care"). Role wording comes from
    ``dfo_platform.roles.role_wording`` for the members' flow family, else
    ``GENERIC_ROLE_WORDING``, and graph role names do not reach the text. A role of more
    than three entities reads as a count, a role asked for as a group
    (``entity_groups``) by the group's size and name, and an open role by its count, or
    "all" when the answer serves the total.

    The noun is a bound what value's label, else the measure label or authored noun with
    each shared flow type's short name, else a graph name (`_resolve_noun`). The how axes
    the noun states (`MeasureNoun.consumed_axes`) are silent. Beside a label or authored
    noun, open axes whose members agree on one value restrict nothing and are silent too.
    With a graph name those values describe the series ("female fertility rate", "person
    ... with income") and speak. A bound how value speaks whatever the slot's observed
    options hold. Those options are a door's sample, the recalled members on a first
    answer and the posted members on a resubmit, so a value read as restricting nothing
    on one door would speak on the other. A unit that changes the magnitude follows the
    noun (`_unit_heading_phrase`). A currency and other units stay out. ``unit_request`` is
    the identity's unit request, whose currency and price basis belong to the byline's
    unit.

    The period is the span the answer serves (`served_period`). The byline states "Latest
    year with data" when ``latest`` marks a request for the latest year, the currency
    with its price basis (`_byline_unit`), and the source, a bound
    source constraint or the resolved provenance.

    ``shape`` needs only ``ShapeLike`` fields. ``facts`` are what the served members give
    (`ResolvedSpec.sentence_facts`), namely one member's name and the flow family.
    ``served`` is the date range the answer serves over them
    (`ResolvedStatVars.data_date_range`), which gives the period. ``facts``, ``served``,
    ``resolved_sources``, ``coverage``, ``scope``, ``facets`` and
    ``role_axis_facts`` fill resolved values into open slots and roles.
    Pre-materialisation callers omit them, leaving their slots silent, the unit absent,
    and open roles unstated. ``keep_how_properties`` preserves sibling-distinguishing
    constraints that a StatVar name or a single value would fold away.
    ``pinned_properties`` names the slots the identity's five-tuple pins, whose one value
    tells no member apart: they and their open-axis facts are silent.
    ``open_axis_facts`` supplies the served members' shared values. ``axis_declarations``
    maps property dcids to code-index names in declaration order. ``measure_label_axes``
    are the partition axes the measure label states, and ``measure_label_named_values``
    the code pairs its authored text names. ``entities`` accepts engine
    ``Entity`` or wire ``ResolvedEntity`` values. ``member_dcids`` are the elected
    members' StatVar dcids, which name the flow family before materialise has served any
    StatVar, so a bind-time heading orders and words its roles as the final one does.
    """
    norm_entities = _normalize_entities(entities)
    constraints = _shared_constraint_values(slots, open_axis_facts)
    family = _flow_family(shape, facts, constraints, member_dcids=member_dcids)
    wording = role_wording(family)
    groups = entity_groups or {}

    noun = _resolve_noun(
        slots,
        shape,
        # A label equal to its dcid is no name (`candidates._graph_name`).
        None if facts is None else [m.label for m in facts.members if m.label != m.ref],
        measure_label,
        measure_label_axes,
        measure_label_named_values,
        constraints,
    )
    modifiers, how = _how_phrases(
        slots,
        noun,
        keep_how_properties=keep_how_properties,
        pinned_properties=pinned_properties,
        open_axis_facts=open_axis_facts,
        axis_declarations=axis_declarations,
        wording=wording,
    )

    where_slot = _find_slot(slots, axis="where")
    roles = _role_phrases(
        norm_entities,
        wording=wording,
        where_slot=where_slot,
        entity_groups=groups,
        role_axis_facts=role_axis_facts,
        scope=scope,
        coverage=coverage,
        # A spoken how value narrows a role's members, so "all" would overstate them.
        states_all=family is not None
        and facts is not None
        and bool(facts.members)
        and not any(value.text for value in (*modifiers, *how)),
    )
    subject = (
        None
        if "destination" in roles
        else _subject_phrase(where_slot, group=groups.get("observationAbout"), coverage=coverage)
    )
    lead: RolePhrase | None = None
    if wording.possessor in roles and roles[wording.possessor].possessable:
        lead = roles.pop(wording.possessor)
    elif not roles and subject is not None and subject.possessable:
        lead, subject = subject, None

    unit_text, basis = _byline_unit(
        shape,
        facets,
        constraints,
        unit_request=unit_request,
        graph_measure_name=None if noun.stated else noun.text,
        wording=wording,
    )
    unit_states = (axis_ref(PRICE_BASIS_AXIS, slots),) if unit_text and basis is not None else ()
    noun = replace(
        noun, states=_noun_states(noun, (*modifiers, *how), slots, unit_states=unit_states)
    )

    extras = tuple(_measure_extras(shape)) if noun.kind == "graph_name" else ()
    # Only a graph name can carry its unit in the name.
    graph_text = (
        None
        if noun.stated
        else " ".join(text for text in (*(v.text for v in modifiers), noun.text, *extras) if text)
    )
    unit_phrase = _unit_heading_phrase(shape, facets, graph_text, wording=wording)
    heading = Heading(
        lead=lead,
        modifiers=tuple(modifiers),
        noun=noun,
        extras=extras,
        unit=unit_phrase.text or None,
        unit_states=(UNIT_REF,) if unit_phrase.text and unit_phrase.names_unit else (),
        roles=tuple(roles.values()),
        how=tuple(how),
        subject=subject,
        period=served_period(served, window),
        when_slot=_find_slot(slots, axis="when"),
        family=family,
    )
    byline = Byline(
        latest=latest,
        unit_text=unit_text or None,
        source=_source_phrase(slots, resolved_sources),
        unit_states=unit_states,
    )
    return HeadingPlan(heading=heading, byline=byline)


def _resolve_noun(
    slots: list[Slot],
    shape: ShapeLike,
    sv_labels: list[str] | None,
    measure_label: str | None,
    measure_label_axes: Sequence[str],
    measure_label_named_values: Sequence[tuple[str, str]],
    constraints: Mapping[str, str],
) -> MeasureNoun:
    """Choose the measure noun and the axes it states.

    Prefer bound what labels, then the engine's ``measure_label``, then the authored
    IDS or WEO noun (`measure_noun`), then ``_fallback_what``. A label or noun takes
    each shared flow type's short name before its head ("gross ODA disbursements") and
    states that flow-type axis, and states each axis whose value it names
    (`_stated_noun`).
    """
    what_slot = _find_slot(slots, axis="what")
    what_labels = _binding_labels(what_slot.binding) if what_slot else []
    if what_labels:
        return MeasureNoun(_join_labels(what_labels), "bound", what_slot)
    if measure_label is not None:
        return _stated_noun(
            measure_label,
            "label",
            what_slot,
            axes=measure_label_axes,
            named_values=measure_label_named_values,
            constraints=constraints,
        )
    authored = _authored_measure_noun(shape, constraints)
    if authored is not None:
        return _stated_noun(
            authored.noun,
            "authored",
            what_slot,
            axes=authored.axes,
            named_values=authored.named_values,
            constraints=constraints,
        )
    text, used_sv_name = _fallback_what(shape, sv_labels, slots)
    return MeasureNoun(text, "stat_var_name" if used_sv_name else "graph_name", what_slot)


def _stated_noun(
    text: str,
    kind: NounKind,
    slot: Slot | None,
    *,
    axes: Sequence[str],
    named_values: Sequence[tuple[str, str]],
    constraints: Mapping[str, str],
) -> MeasureNoun:
    """A measure label or authored noun with the shared flow type's short name, and the
    axes it states.

    The short name goes before the noun's head (`_with_short_names`) unless the noun
    names that flow type itself. Either way the flow-type axis joins the noun's
    ``measure_axes``. Each of the noun's ``named_values`` that the members share names
    its property (`MeasureNoun.named_axes`).
    """
    named = [prop for prop, value in named_values if constraints.get(prop) == value]
    short_names = _flow_short_names(constraints)
    measure_axes = (*axes, *(prop for prop, _ in short_names))
    return MeasureNoun(
        _with_short_names(text, [name for prop, name in short_names if prop not in named]),
        kind,
        slot,
        measure_axes,
        tuple(prop for prop in named if prop not in measure_axes),
    )


def _noun_states(
    noun: MeasureNoun,
    values: Sequence[HowValue],
    slots: Sequence[Slot],
    *,
    unit_states: Sequence[str],
) -> tuple[str, ...]:
    """The refs the noun's token states besides its own slot: each how value silent
    because the noun says it (`HowValue.named_by_noun`), then each axis the noun states
    (`MeasureNoun.consumed_axes`, `axis_ref`). A ref the unit token states
    (``unit_states``) is the unit's."""
    refs = [value.slot.key.as_ref() for value in values if value.named_by_noun and value.slot]
    refs.extend(axis_ref(prop, slots) for prop in (*noun.measure_axes, *noun.named_axes))
    own = noun.slot.key.as_ref() if noun.slot is not None else None
    return tuple(ref for ref in dict.fromkeys(refs) if ref != own and ref not in unit_states)


def _how_phrases(
    slots: list[Slot],
    noun: MeasureNoun,
    *,
    keep_how_properties: frozenset[str],
    pinned_properties: frozenset[str],
    open_axis_facts: Sequence[OpenConstraintAxisFact],
    axis_declarations: Mapping[str, str] | None,
    wording: RoleWording,
) -> tuple[list[HowValue], list[HowValue]]:
    """The modifiers before the noun and the how values after the roles (`how_values`).

    A how slot on one of the noun's ``measure_axes``, or one the identity's five-tuple pins
    (``pinned_properties``), is silent and follows the other how values. One on the noun's
    ``named_axes`` is silent in its place. Open-axis facts on a
    consumed axis say nothing. Beside a label or authored noun, the values the members
    share restrict nothing, so open-axis facts stay silent unless ``keep_how_properties``
    names them, and the noun states their slots (``settled_axes``). A StatVar name folds
    other bound how slots unless ``keep_how_properties`` protects them.
    """
    measure_axes = frozenset(noun.measure_axes) | pinned_properties
    consumed = noun.consumed_axes | pinned_properties
    settled = frozenset(
        fact.property
        for fact in open_axis_facts
        if noun.stated
        and fact.property not in consumed
        and fact.property not in keep_how_properties
        and not fact.varies
        and fact.value is not None
    )
    walked: list[Slot] = []
    measure_slots: list[Slot] = []
    for slot in slots:
        prop_dcid = slot.key.property.ref if slot.key.property else None
        if slot.key.axis == "how" and prop_dcid in measure_axes:
            measure_slots.append(slot)
        else:
            walked.append(slot)
    modifiers, how = how_values(
        walked,
        graph_name=None if noun.stated else noun.text,
        used_sv_name=noun.kind == "stat_var_name",
        keep_how_properties=keep_how_properties,
        open_axis_facts=[
            fact
            for fact in open_axis_facts
            if fact.property not in consumed
            and (not noun.stated or fact.property in keep_how_properties)
        ],
        axis_declarations=axis_declarations,
        named_axes=frozenset(noun.named_axes),
        settled_axes=settled,
        wording=wording,
    )
    how.extend(
        HowValue(
            slot.key.property.ref if slot.key.property else None, "", "", slot, named_by_noun=True
        )
        for slot in measure_slots
    )
    return modifiers, how


# More entities than this in one role read as a count ("27 donors").
_MAX_NAMED_ENTITIES = 3

# Flow roles in heading order after the noun: "to Kenya from Germany".
_ROLE_ORDER: tuple[RoleAxis, ...] = ("destination", "origin")

# Coverage dimension labels that count an open role's entities, by the role they count.
_ROLE_COVERAGE_LABELS: Mapping[str, RoleAxis] = {
    "recipients": "destination",
    "providers": "origin",
}


def _entities_text(labels: list[str], *, group: EntityGroup | None, plural: str) -> str:
    """Name a role's entities: by group and count (`group_text`), by count past three,
    else by name."""
    if group is not None and len(labels) > 1:
        return group_text(len(labels), group)
    if len(labels) > _MAX_NAMED_ENTITIES:
        return f"{len(labels)} {plural}"
    return _join_labels(labels)


def _open_role_reading(
    axis: RoleAxis,
    noun: RoleNoun,
    *,
    role_axis_facts: Sequence[RoleAxisFact],
    scope: ResolvedScope | None,
    coverage: Coverage | None,
    states_all: bool,
) -> tuple[str, list[_NormalizedEntity]] | None:
    """Return the text and entities naming an open role, or None when nothing is known.

    A scope of one member on that axis names that member. Otherwise the count comes from
    the role's `RoleAxisFact`, then the scope, then a coverage dimension counting it, and
    reads "{n} {plural}". A count of one with no name says nothing, so it reads nothing.
    With no count, ``states_all`` reads the role as every entity in it ("all creditors").
    """
    members = list(scope.members) if scope is not None and scope.axis == axis else []
    if len(members) == 1:
        member = members[0]
        entity = _NormalizedEntity(dcid=member.ref, label=member.label, axis=axis)
        return _entity_name(member.ref, member.label), [entity]
    counts = [fact.member_count for fact in role_axis_facts if fact.role == axis]
    if members:
        counts.append(len(members))
    if isinstance(coverage, (CoverageExact, CoverageBreadth)):
        counts.extend(
            dim.count
            for dim in coverage.dimensions or ()
            if _ROLE_COVERAGE_LABELS.get(dim.label) == axis
        )
    count = next((count for count in counts if count is not None), None)
    if count is not None:
        return (f"{count} {noun.plural}", []) if count > 1 else None
    if states_all:
        return f"all {noun.plural}", []
    return None


def _role_phrases(
    norm_entities: list[_NormalizedEntity],
    *,
    wording: RoleWording,
    where_slot: Slot | None,
    entity_groups: Mapping[str, EntityGroup],
    role_axis_facts: Sequence[RoleAxisFact],
    scope: ResolvedScope | None,
    coverage: Coverage | None,
    states_all: bool,
) -> dict[RoleAxis, RolePhrase]:
    """Return each flow role's phrase, destination first, with the family's preposition.

    A role with entities names them (`_entities_text`). An open role reads its count or
    "all" (`_open_role_reading`). The destination carries the where slot, which it shares
    with the subject.
    """
    phrases: dict[RoleAxis, RolePhrase] = {}
    for axis in _ROLE_ORDER:
        noun = wording.noun(axis)
        entities = [entity for entity in norm_entities if entity.axis == axis]
        group: EntityGroup | None = None
        if entities:
            group = entity_groups.get(axis) if len(entities) > 1 else None
            labels = [_entity_name(e.dcid, e.label) for e in entities]
            text = _entities_text(labels, group=group, plural=noun.plural)
            possessable = len(entities) == 1 and axis not in entity_groups
        else:
            reading = _open_role_reading(
                axis,
                noun,
                role_axis_facts=role_axis_facts,
                scope=scope,
                coverage=coverage,
                states_all=states_all,
            )
            if reading is None:
                continue
            text, entities = reading
            possessable = False
        phrases[axis] = RolePhrase(
            axis=axis,
            preposition=noun.preposition,
            text=text,
            entity_dcids=tuple(e.dcid for e in entities),
            slot=where_slot if axis == "destination" else None,
            possessable=possessable,
        )
    return phrases


def _subject_phrase(
    where_slot: Slot | None,
    *,
    group: EntityGroup | None,
    coverage: Coverage | None,
) -> RolePhrase | None:
    """Return the where slot's phrase for a heading with no destination, or None.

    Bound places read like a role's entities, with "in", and one place can lead the
    heading in possessive form. An unbound slot reads coverage ("worldwide", "across 78
    countries") as body text and holds its place with empty text otherwise.
    """
    if where_slot is None:
        return None
    labels = _entity_labels(where_slot.binding)
    if labels:
        dcids = tuple(v.ref.ref for v in _binding_values(where_slot.binding) if v.ref)
        return RolePhrase(
            axis=None,
            preposition="in",
            text=_entities_text(labels, group=group, plural="places"),
            entity_dcids=dcids,
            slot=where_slot,
            possessable=len(labels) == 1 and group is None,
        )
    return RolePhrase(
        axis=None,
        preposition=None,
        text=_resolved_where_text(coverage),
        entity_dcids=(),
        slot=where_slot,
        possessable=False,
    )


def _flow_family(
    shape: ShapeLike,
    facts: SentenceFacts | None,
    constraints: Mapping[str, str],
    *,
    member_dcids: Sequence[str] = (),
) -> FlowFamily | None:
    """The flow family the members share (`flow_family`), or None.

    The members are the elected ``member_dcids`` when known, else the served members
    (`SentenceFacts.members`). For StatVars outside the index, each constraint property
    list the served members state is one member, valued from the values every member
    shares. With none, the shared values stand for the one member.
    """
    stat_vars = list(member_dcids) or [
        member.ref for member in (facts.members if facts is not None else ())
    ]
    members = [
        {prop_dcid: constraints.get(prop_dcid, "") for prop_dcid in properties}
        for properties in (facts.constraint_properties if facts is not None else ())
    ]
    if not members and constraints:
        members = [dict(constraints)]
    population_type = shape.population_type.ref if shape.population_type else None
    return flow_family(
        index=get_index(),
        population_type=population_type,
        stat_vars=stat_vars,
        members=members,
    )


def _byline_unit(
    shape: ShapeLike,
    facets: Mapping[str, ResolvedFacet] | None,
    constraints: Mapping[str, str],
    *,
    unit_request: IdentityUnitRequest | None,
    graph_measure_name: str | None,
    wording: RoleWording,
) -> tuple[str, PriceBasis | None]:
    """The byline's currency with its price basis, or empty when the byline states none,
    and the price basis it names, None when it names none.

    Every currency reads here, by its plural noun (`currency_noun`): "current US dollars",
    "constant 2024 euros", "current Guatemalan quetzales" (`price_basis_phrase`). The unit
    facet's ISO code (`ResolvedFacet.currency`) says which currency it is. A unit request
    for a role's currency names no single unit facet, so it reads as that role's currency
    ("constant 2024 donor currency", `_role_currency_noun`). Other units are stated by
    the heading when they change the magnitude (`_unit_heading_phrase`) or needed by no
    reader. A unit whose series disagree (``per_series``) reads nothing. The price basis
    is the unit request's, the unit's base year, or the ``measurementQualifier`` of the
    shape or of every member (`price_basis`).

    ``graph_measure_name`` is the graph name the heading states for the measure, None
    when an authored noun or measure label states it. Only a graph name is read as text.
    A name that already states the unit's currency ("gross domestic product at current
    prices in US dollars", `_names_currency`) leaves the byline without a unit, so it
    reads once.
    """
    unit = (facets or {}).get("unit")
    unit_class = None
    if unit is not None and unit.ref is not None:
        denominator = shape.measurement_denominator
        unit_class = classify_unit(
            unit=unit.ref,
            denominator=denominator.ref if denominator is not None else None,
            currency_units=facet_currency_units(unit),
        )
    if unit is not None and unit_class is not None and unit_class.currency is not None:
        if graph_measure_name is not None and _names_currency(
            graph_measure_name, unit_class.currency
        ):
            return "", None
        noun = currency_noun(unit_class.currency, label=unit.label)
    elif unit_request is not None and isinstance(unit_request.currency, IdentityCurrencyRole):
        noun = _role_currency_noun(unit_request.currency, wording)
    else:
        return "", None
    qualifier = shape.measurement_qualifier
    basis = price_basis(
        requested=(
            PriceBasis(prices=unit_request.prices, base_year=unit_request.base_year)
            if unit_request is not None
            else None
        ),
        unit_class=unit_class,
        measurement_qualifier=(
            qualifier.ref if qualifier is not None else constraints.get(PRICE_BASIS_AXIS)
        ),
    )
    return price_basis_phrase(basis, currency_noun=noun), basis


def _role_currency_noun(currency: IdentityCurrencyRole, wording: RoleWording) -> str:
    """The noun of a role's currency: "donor currency", "recipient currency", or
    "national currency" for the subject."""
    if currency.role == "observationAbout":
        return "national currency"
    return f"{wording.noun(currency.role).singular} currency"


def _source_phrase(
    slots: list[Slot], resolved_sources: Sequence[GraphRef] | None
) -> SourcePhrase | None:
    """The source: a bound source constraint, else the answer's resolved provenance,
    which no slot backs because source options are not published. An unbound source
    slot with no provenance holds its place with empty text. No slot and no provenance
    give None."""
    source_slot = _find_slot(slots, axis="source")
    source_labels = _binding_labels(source_slot.binding) if source_slot else []
    if source_labels:
        # Humanise before joining because source-identifier matching uses the full label.
        text = _join_labels([_humanize_source_label(label) for label in source_labels])
        return SourcePhrase(text, source_slot)
    resolved_source = _resolved_source_text(resolved_sources)
    if resolved_source:
        return SourcePhrase(resolved_source, None)
    if source_slot is not None:
        return SourcePhrase("", source_slot)
    return None


def _find_slot(slots: list[Slot], *, axis: str) -> Slot | None:
    """The first Slot on `axis`, or None when the shape declares no such slot."""
    for slot in slots:
        if slot.key.axis == axis:
            return slot
    return None


def _resolved_where_text(coverage: Coverage | None) -> str:
    """Return the resolved place scope label, or empty when unavailable.

    Read ``worldwide``, ``places``, or ``countries`` from exact or breadth coverage. A
    role's count reads through `_open_role_reading`. ``worldwide`` renders as-is. Other
    labels use "across N <label>" when count exceeds one. Omit count one because
    coverage carries no entity name. Other coverage types and dimensions produce no text.
    """
    if coverage is None or not isinstance(coverage, (CoverageExact, CoverageBreadth)):
        return ""
    if not coverage.dimensions:
        return ""
    for dim in coverage.dimensions:
        if dim.label == "worldwide":
            return "worldwide"
        if dim.label in ("places", "countries"):
            return f"across {dim.count} {dim.label}" if dim.count != 1 else ""
    return ""


def _resolved_source_text(resolved_sources: Sequence[GraphRef] | None) -> str:
    """Render one to three resolved sources by name, or four or more as a count.

    Humanise each source before joining so raw import identifiers are split
    individually. No sources returns empty text.
    """
    if not resolved_sources:
        return ""
    if len(resolved_sources) >= 4:
        return f"{len(resolved_sources)} sources"
    return _join_labels([_humanize_source_label(_humanize(ref)) for ref in resolved_sources])
