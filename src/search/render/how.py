"""Render bound and open how-axis values, including adjectival folds and redundancy checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from code_index.labels import (
    HOW_CONNECTIVES,
    connective_for_phrase,
    how_value_text,
    role_class_text,
)
from code_index.shared import get_index
from dfo_platform.roles import GENERIC_ROLE_WORDING, RoleWording
from search.models import OpenConstraintAxisFact, Slot, SlotValue

from .labels import (
    _already_named_in,
    _binding_values,
    _humanize_ref,
    join_labels,
)

# A public-sector debtor's debt is public, so "public or publicly guaranteed" says nothing
# beside one. The graph says which debtor types are public-sector: `IDSDebtorTypePS` and the
# values whose `specializationOf` chain reaches it (`CodeIndex.value_ancestors`).
_DEBTOR_TYPE = "idsDebtorType"
_PUBLIC_SECTOR_DEBTOR = "IDSDebtorTypePS"
_GUARANTEE_STATUS = "guaranteeStatus"
_PUBLIC_OR_PUBLICLY_GUARANTEED = "PublicOrPubliclyGuaranteed"


def _implied_properties(
    slots: list[Slot], open_axis_facts: Sequence[OpenConstraintAxisFact]
) -> frozenset[str]:
    """The how properties whose one shared value another shared value implies, so the
    sentence omits it: the guarantee status "public or publicly guaranteed" beside a
    public-sector debtor type."""
    shared = _shared_constraint_values(slots, open_axis_facts)
    debtor = shared.get(_DEBTOR_TYPE)
    if debtor is None or shared.get(_GUARANTEE_STATUS) != _PUBLIC_OR_PUBLICLY_GUARANTEED:
        return frozenset()
    public = debtor == _PUBLIC_SECTOR_DEBTOR or _PUBLIC_SECTOR_DEBTOR in (
        get_index().value_ancestors(debtor)
    )
    return frozenset({_GUARANTEE_STATUS}) if public else frozenset()


def _is_adjectival(slot: Slot) -> bool:
    """Whether a how-slot's constraint property reads as a pre-noun modifier."""
    prop_dcid = slot.key.property.ref if slot.key.property else None
    return prop_dcid is not None and HOW_CONNECTIVES.get(prop_dcid) == ""


@dataclass(frozen=True, slots=True)
class HowValue:
    """One how value a heading states, or a silent slot holding its place.

    ``text`` is the value's phrase, empty for a slot that states nothing and keeps its
    wire-visible token. ``connective`` is the text before a value that follows the noun
    (" for ", " as ", or " " when the phrase opens with its own preposition), empty on a
    modifier before the noun and on a silent value. ``slot`` backs the value. A value an
    open axis alone settles has none and carries its ``property``. ``named_by_noun``
    marks a silent value the measure noun says instead, so the noun's token states its
    slot (`SentenceToken.states`).
    """

    property: str | None
    text: str
    connective: str
    slot: Slot | None
    named_by_noun: bool = False


def how_values(
    slots: list[Slot],
    *,
    graph_name: str | None,
    used_sv_name: bool = False,
    keep_how_properties: frozenset[str] = frozenset(),
    open_axis_facts: Sequence[OpenConstraintAxisFact] = (),
    axis_declarations: Mapping[str, str] | None = None,
    named_axes: frozenset[str] = frozenset(),
    settled_axes: frozenset[str] = frozenset(),
    wording: RoleWording = GENERIC_ROLE_WORDING,
) -> tuple[list[HowValue], list[HowValue]]:
    """Split how-axis values into pre-noun modifiers and post-noun values.

    Bound adjectival slots go before the noun in slot order. Other bound slots and
    unbound placeholders follow it. ``used_sv_name`` folds bound how slots to silent
    values because the StatVar name already names their constraints.
    ``keep_how_properties`` preserves sibling differences and bypasses boolean
    suppression. A slot whose property is in ``named_axes`` is silent in its place,
    since the measure noun is named after its value (`MeasureNoun.named_axes`).
    ``settled_axes`` are the open axes whose agreed value a label or authored noun stands
    for. Their facts reach no value here, and their unbound slots are the noun's.

    ``graph_name`` is the measure's graph name when one states the measure, None when an
    authored noun or measure label does. Only a graph name is read as text, and a value
    label it already contains (`_already_named_in`) is omitted.

    Process declared slots and ``open_axis_facts`` together. An open fact with one
    agreed value states that humanised value, unless ``used_sv_name`` folds it as it
    folds a bound slot. Varying or valueless facts stay silent. A slot-backed silence
    keeps a silent value with its slot; fact-only silence has none. Post-noun values
    sort bound values before fact-derived values, then by declaration order and property
    dcid. Modifiers retain slot order.

    Boolean members are omitted, except kept booleans. Adjectival folds disappear if any
    member is omitted. Post-noun values keep and join surviving members. Each property's
    connective is declared once (`how_connective`). A silent value the noun says instead
    (a named or settled axis, a StatVar name's fold, or every member already in the
    graph name) is marked `HowValue.named_by_noun`. A value of a role-class property
    reads as that role's entities in ``wording`` ("owed to bilateral creditors",
    `role_class_text`). A value label ending in its property's name in brackets
    ("bilateral (IDS creditor type)") reads without it. A value another shared value
    implies (`_implied_properties`) is silent in its place unless ``keep_how_properties``
    names its property.
    """
    declarations = axis_declarations or {}
    order_index = {dcid: i for i, dcid in enumerate(declarations)}
    facts_by_property = {fact.property: fact for fact in open_axis_facts}
    how_slots = [slot for slot in slots if slot.key.axis == "how"]
    slots_by_property: dict[str | None, Slot] = {
        (slot.key.property.ref if slot.key.property else None): slot for slot in how_slots
    }
    implied = _implied_properties(slots, open_axis_facts) - keep_how_properties

    pre: list[HowValue] = []
    # Sort post-noun values by origin, declaration order, then property dcid.
    entries: list[tuple[tuple[int, int, str], HowValue]] = []
    unresolved_props: list[str | None] = []

    for slot in how_slots:
        prop_dcid = slot.key.property.ref if slot.key.property else None
        values = _binding_values(slot.binding)
        if not values:
            unresolved_props.append(prop_dcid)
            continue
        is_kept = prop_dcid in keep_how_properties
        silent = prop_dcid in named_axes or (used_sv_name and not is_kept)
        property_label = slot.key.property.label if slot.key.property else None
        sort_key = (0, order_index.get(prop_dcid, len(order_index)), prop_dcid or "")
        if prop_dcid in implied and not silent:
            entries.append((sort_key, HowValue(prop_dcid, "", "", slot)))
            continue
        if _is_adjectival(slot):
            text = (
                "" if silent else _adjectival_fold_text(values, graph_name, bypass_boolean=is_kept)
            )
            named = silent or (
                not text and _named_by_graph_name(values, graph_name, bypass_boolean=is_kept)
            )
            pre.append(HowValue(prop_dcid, text, "", slot, named_by_noun=named))
            continue
        value = HowValue(prop_dcid, "", "", slot, named_by_noun=silent)
        if not silent:
            member_texts = [
                role_class_text(text, prop_dcid, wording)
                for v in values
                if (
                    text := _how_member_text(
                        v,
                        graph_name,
                        bypass_boolean=is_kept,
                        prop_dcid=prop_dcid,
                        property_label=property_label,
                    )
                )
                is not None
            ]
            if member_texts:
                phrase = join_labels(member_texts)
                connective = connective_for_phrase(prop_dcid, phrase, wording)
                value = HowValue(prop_dcid, phrase, connective, slot)
            else:
                named = _named_by_graph_name(
                    values,
                    graph_name,
                    bypass_boolean=is_kept,
                    prop_dcid=prop_dcid,
                    property_label=property_label,
                )
                value = HowValue(prop_dcid, "", "", slot, named_by_noun=named)
        entries.append((sort_key, value))

    # Include properties represented only by open-axis facts.
    for prop_dcid in facts_by_property:
        if prop_dcid not in slots_by_property and prop_dcid not in unresolved_props:
            unresolved_props.append(prop_dcid)

    for prop_dcid in unresolved_props:
        slot = slots_by_property.get(prop_dcid)
        fact = facts_by_property.get(prop_dcid) if prop_dcid is not None else None
        sort_key = (1, order_index.get(prop_dcid, len(order_index)), prop_dcid or "")
        named_by_stat_var = used_sv_name and prop_dcid not in keep_how_properties
        agreed = fact is not None and not fact.varies and fact.value is not None
        if prop_dcid in implied and not named_by_stat_var:
            if slot is not None:
                entries.append((sort_key, HowValue(prop_dcid, "", "", slot)))
            continue
        if (
            fact is not None
            and not fact.varies
            and fact.value is not None
            and not named_by_stat_var
        ):
            property_label = slot.key.property.label if slot and slot.key.property else None
            value_text = role_class_text(
                _how_value_text(fact.value.ref, fact.value.label, prop_dcid, property_label),
                prop_dcid,
                wording,
            )
            connective = connective_for_phrase(prop_dcid, value_text, wording)
            entries.append((sort_key, HowValue(prop_dcid, value_text, connective, slot)))
        elif slot is not None:
            named = (
                prop_dcid in named_axes
                or prop_dcid in settled_axes
                or (agreed and named_by_stat_var)
            )
            entries.append((sort_key, HowValue(prop_dcid, "", "", slot, named_by_noun=named)))
        # Fact-only properties without an agreed value state nothing.

    entries.sort(key=lambda item: item[0])
    return pre, [value for _, value in entries]


# Boolean literal or dcid-tail values are omitted case-insensitively.
_BOOLEAN_MEMBER_TEXT = frozenset({"true", "false"})


def _is_boolean_member(value: SlotValue) -> bool:
    """Return whether a literal or dcid tail is "true" or "false", case-insensitively."""
    if value.literal is not None:
        return value.literal.strip().lower() in _BOOLEAN_MEMBER_TEXT
    if value.ref is not None:
        return value.ref.ref.rsplit("/", 1)[-1].lower() in _BOOLEAN_MEMBER_TEXT
    return False


def _how_member_text(
    value: SlotValue,
    graph_name: str | None,
    *,
    bypass_boolean: bool = False,
    prop_dcid: str | None = None,
    property_label: str | None = None,
) -> str | None:
    """Return a member's text (`_how_value_text`), or None when it should be omitted.

    Boolean members are omitted unless ``bypass_boolean`` is true. Labels already in
    the measure's ``graph_name`` are always omitted. An authored noun or measure label
    (``graph_name`` None) is not read. Kept properties bypass the boolean filter to
    preserve sibling differences. Adjectival callers discard the whole fold if any
    member returns None. Post-noun callers keep surviving labels.
    """
    if not bypass_boolean and _is_boolean_member(value):
        return None
    if value.ref is not None:
        label = _how_value_text(value.ref.ref, value.ref.label, prop_dcid, property_label)
    elif value.literal:
        label = value.literal
    else:
        return None
    return None if graph_name is not None and _already_named_in(label, graph_name) else label


def _named_by_graph_name(
    values: list[SlotValue],
    graph_name: str | None,
    *,
    bypass_boolean: bool,
    prop_dcid: str | None = None,
    property_label: str | None = None,
) -> bool:
    """Whether the graph name already names every member (`_already_named_in`), so the
    noun says the whole value. A boolean member the filter drops is named by nothing."""
    if graph_name is None or not values:
        return False
    for value in values:
        label = _how_member_text(
            value,
            None,
            bypass_boolean=bypass_boolean,
            prop_dcid=prop_dcid,
            property_label=property_label,
        )
        if label is None or not _already_named_in(label, graph_name):
            return False
    return True


def _how_value_text(
    dcid: str, label: str | None, prop_dcid: str | None, property_label: str | None
) -> str:
    """A how value's text (`code_index.labels.how_value_text`), over its label or, without
    one, its humanised dcid."""
    return how_value_text(
        dcid,
        _humanize_ref(dcid, label),
        prop=prop_dcid,
        property_label=property_label,
        index=get_index(),
    )


def _adjectival_fold_text(
    values: list[SlotValue], graph_name: str | None, *, bypass_boolean: bool = False
) -> str:
    """Join all labels. Return empty if any member drops, avoiding partial modifiers."""
    labels: list[str] = []
    for value in values:
        text = _how_member_text(value, graph_name, bypass_boolean=bypass_boolean)
        if text is None:
            return ""
        labels.append(text)
    return join_labels(labels)


def _shared_constraint_values(
    slots: list[Slot], open_axis_facts: Sequence[OpenConstraintAxisFact]
) -> dict[str, str]:
    """Return the constraint values every served member shares, property dcid to value dcid.

    A how slot bound to one graph value, and an open axis whose members agree on one
    value, each give one entry. A property bound to several values or varying across
    members is left out.
    """
    shared: dict[str, str] = {}
    for slot in slots:
        values = _binding_values(slot.binding)
        if slot.key.axis == "how" and slot.key.property and len(values) == 1:
            ref = values[0].ref
            if ref is not None:
                shared[slot.key.property.ref] = ref.ref
    for fact in open_axis_facts:
        if not fact.varies and fact.value is not None:
            shared.setdefault(fact.property, fact.value.ref)
    return shared
