"""Render bound and open how-axis values, including adjectival folds and redundancy checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from code_index.shared import get_index
from dfo_platform.roles import GENERIC_ROLE_WORDING, RoleAxis, RoleWording
from search.models import OpenConstraintAxisFact, Slot, SlotValue

from .labels import (
    _already_named_in,
    _binding_values,
    _humanize_ref,
    _join_labels,
)

_ADJECTIVAL_PROPERTIES = (
    "gender",
    "age",
    "race",
    "maritalStatus",
    "nativity",
    "residenceType",
    "veteranStatus",
    "armedForcesStatus",
    "employment",
    "educationalAttainment",
    "povertyStatus",
    "healthInsurance",
    "disabilityStatus",
)
_PLACE_CATEGORY_PROPERTIES = (
    "placeOfBirth",
    "placeOfResidenceClassification",
    "placeOfWork",
    "placeCategory",
    "locationType",
    "jurisdiction",
    "computerUsageLocation",
    "internetUsageLocation",
)

_HOW_CONNECTIVES: Mapping[str, str] = {
    **dict.fromkeys(_ADJECTIVAL_PROPERTIES, ""),
    **dict.fromkeys(_PLACE_CATEGORY_PROPERTIES, " of "),
    "crsPurpose": " for ",
    "crsTypeOfFinance": " as ",
    "crsCooperationModality": " through ",
}
"""The connective before a bound or agreed how value, by constraint property, with its
surrounding spaces. Demographic properties modify the noun directly (""), place
categories read "of", a sector "for", a finance type "as" ("as grants"), and a
cooperation modality "through" ("through sector budget support"). A property not
listed reads `_DEFAULT_HOW_CONNECTIVE`. One in `_ROLE_CLASS_PROPERTIES` reads its role's
preposition."""

_DEFAULT_HOW_CONNECTIVE = " for "

_ROLE_CLASS_PROPERTIES: Mapping[str, RoleAxis] = {"idsCreditorType": "origin"}
"""How properties whose values name a class of one flow role's entities. The graph
describes `idsCreditorType` as "the class of lender an amount is owed to", so its value
reads as the creditor role in the family's wording: "owed to bilateral creditors"."""


def _how_connective(prop_dcid: str | None, wording: RoleWording) -> str:
    """Return a property's connective with its surrounding spaces (`_HOW_CONNECTIVES`)."""
    if prop_dcid is None:
        return _DEFAULT_HOW_CONNECTIVE
    if (axis := _ROLE_CLASS_PROPERTIES.get(prop_dcid)) is not None:
        return f" {wording.noun(axis).preposition} "
    return _HOW_CONNECTIVES.get(prop_dcid, _DEFAULT_HOW_CONNECTIVE)


def _is_adjectival(slot: Slot) -> bool:
    """Whether a how-slot's constraint property reads as a pre-noun modifier."""
    prop_dcid = slot.key.property.dcid if slot.key.property else None
    return prop_dcid is not None and _HOW_CONNECTIVES.get(prop_dcid) == ""


def _role_class_text(label: str, prop_dcid: str | None, wording: RoleWording) -> str:
    """Name a role-class value as that role's entities: the label, then the role's plural
    noun unless the label already ends in a plural ("bilateral" reads "bilateral
    creditors", "bondholders" stays). Matching the final "s" is a text rule. The ten IDS
    creditor-type labels it reads are pinned in tests/render/test_render_heading_words.py."""
    axis = _ROLE_CLASS_PROPERTIES.get(prop_dcid) if prop_dcid is not None else None
    if axis is None or label.endswith("s"):
        return label
    return f"{label} {wording.noun(axis).plural}"


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
    connective is declared once (`_how_connective`). A silent value the noun says instead
    (a named or settled axis, a StatVar name's fold, or every member already in the
    graph name) is marked `HowValue.named_by_noun`. A value of a role-class property
    reads as that role's entities in ``wording`` ("owed to bilateral creditors",
    `_role_class_text`). A value label ending in its property's name in brackets
    ("bilateral (IDS creditor type)") reads without it.
    """
    declarations = axis_declarations or {}
    order_index = {dcid: i for i, dcid in enumerate(declarations)}
    facts_by_property = {fact.property: fact for fact in open_axis_facts}
    how_slots = [slot for slot in slots if slot.key.axis == "how"]
    slots_by_property: dict[str | None, Slot] = {
        (slot.key.property.dcid if slot.key.property else None): slot for slot in how_slots
    }

    pre: list[HowValue] = []
    # Sort post-noun values by origin, declaration order, then property dcid.
    entries: list[tuple[tuple[int, int, str], HowValue]] = []
    unresolved_props: list[str | None] = []

    for slot in how_slots:
        prop_dcid = slot.key.property.dcid if slot.key.property else None
        values = _binding_values(slot.binding)
        if not values:
            unresolved_props.append(prop_dcid)
            continue
        is_kept = prop_dcid in keep_how_properties
        silent = prop_dcid in named_axes or (used_sv_name and not is_kept)
        property_label = slot.key.property.label if slot.key.property else None
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
                _role_class_text(text, prop_dcid, wording)
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
                phrase = _join_labels(member_texts)
                connective = _connective_for_phrase(prop_dcid, phrase, wording)
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
        entries.append(((0, order_index.get(prop_dcid, len(order_index)), prop_dcid or ""), value))

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
        if (
            fact is not None
            and not fact.varies
            and fact.value is not None
            and not named_by_stat_var
        ):
            property_label = slot.key.property.label if slot and slot.key.property else None
            value_text = _role_class_text(
                _how_value_text(fact.value.dcid, fact.value.label, prop_dcid, property_label),
                prop_dcid,
                wording,
            )
            connective = _connective_for_phrase(prop_dcid, value_text, wording)
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
        return value.ref.dcid.rsplit("/", 1)[-1].lower() in _BOOLEAN_MEMBER_TEXT
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
        label = _how_value_text(value.ref.dcid, value.ref.label, prop_dcid, property_label)
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
    """A how value's text: its code's short name (`CodeRow.short_name`, "grants" for
    `CRSTypeOfFinance100`) when its code row belongs to ``prop_dcid``, else its label
    without the property suffix (`_without_property_suffix`)."""
    row = get_index().code_by_dcid(dcid)
    if row is not None and row.short_name and row.code_system == prop_dcid:
        return row.short_name
    return _without_property_suffix(_humanize_ref(dcid, label), property_label)


def _without_property_suffix(label: str, property_label: str | None) -> str:
    """Drop a trailing bracketed property name the graph adds to tell code lists apart:
    "bilateral (IDS creditor type)" reads "bilateral"."""
    if not property_label:
        return label
    suffix = f" ({property_label})"
    return label[: -len(suffix)] if label.lower().endswith(suffix.lower()) else label


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
    return _join_labels(labels)


# Avoid doubling a preposition already present in the value phrase.
_HOW_PHRASE_LEADS_WITH_PREPOSITION = frozenset(
    {
        "with",
        "without",
        "in",
        "on",
        "at",
        "by",
        "for",
        "from",
        "of",
        "to",
        "under",
        "over",
        "among",
        "per",
    }
)


def _connective_for_phrase(prop_dcid: str | None, phrase: str, wording: RoleWording) -> str:
    """Return the connective for a value phrase after the noun, or a bare space when the
    phrase starts with a preposition. An adjectival property's value that follows the
    noun (an open axis's agreed value) reads `_DEFAULT_HOW_CONNECTIVE`."""
    first_word = phrase.split(" ", 1)[0].lower()
    if first_word in _HOW_PHRASE_LEADS_WITH_PREPOSITION:
        return " "
    return _how_connective(prop_dcid, wording) or _DEFAULT_HOW_CONNECTIVE


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
                shared[slot.key.property.dcid] = ref.dcid
    for fact in open_axis_facts:
        if not fact.varies and fact.value is not None:
            shared.setdefault(fact.property, fact.value.dcid)
    return shared
