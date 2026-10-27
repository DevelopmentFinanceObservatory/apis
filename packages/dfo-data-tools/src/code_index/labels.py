"""Labels the release's code indexes override: a measure partition, a StatVar family, an index
family, and a node's display label."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Literal, cast

from code_index.composites import composite_ref, without_composite
from code_index.members import elected_pins, identity_family_query
from code_index.query import CodeIndex, IndexFamily
from code_index.types import MeasureIndex, MeasureValueRow, PartitionLabelMatch, label_for_pairs
from dfo_platform.roles import RoleAxis, RoleWording

PRICE_BASIS_AXIS = "measurementQualifier"
"""The measure axis holding a StatVar's native price basis (`Nominal`). Unit conversion reads
it as the price basis it converts from."""

PricesKind = Literal["current", "constant"]
"""Current (nominal) or constant (real) prices."""

PRICES_BY_QUALIFIER: Mapping[str, PricesKind] = MappingProxyType(
    {"Nominal": "current", "RealValue": "constant"}
)
"""The `PRICE_BASIS_AXIS` values that state a price basis, and the prices each states. Another
value on that axis (`Adjusted`) is no price basis. No measure label states a price basis,
because the answer's unit phrase does."""


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

HOW_CONNECTIVES: Mapping[str, str] = MappingProxyType(
    {
        **dict.fromkeys(_ADJECTIVAL_PROPERTIES, ""),
        **dict.fromkeys(_PLACE_CATEGORY_PROPERTIES, " of "),
        "crsPurpose": " for ",
        "crsTypeOfFinance": " as ",
        "crsCooperationModality": " through ",
    }
)
"""The connective before a bound or agreed how value, by constraint property, with its
surrounding spaces. Demographic properties modify the noun directly (""), place
categories read "of", a sector "for", a finance type "as" ("as grants"), and a
cooperation modality "through" ("through sector budget support"). A property not
listed reads `DEFAULT_HOW_CONNECTIVE`, and one in `ROLE_CLASS_PROPERTIES` its role's
preposition (`how_connective`). Search's heading and the description composer's title read
it."""

DEFAULT_HOW_CONNECTIVE = " for "
"""The connective before a how value whose property `HOW_CONNECTIVES` does not list."""


ROLE_CLASS_PROPERTIES: Mapping[str, RoleAxis] = MappingProxyType({"idsCreditorType": "origin"})
"""How properties whose values name a class of one flow role's entities. The graph
describes `idsCreditorType` as "the class of lender an amount is owed to", so its value
reads as the creditor role in the family's wording: "owed to bilateral creditors"."""


def how_connective(prop_dcid: str | None, wording: RoleWording) -> str:
    """Return a property's connective with its surrounding spaces (`HOW_CONNECTIVES`)."""
    if prop_dcid is None:
        return DEFAULT_HOW_CONNECTIVE
    if (axis := ROLE_CLASS_PROPERTIES.get(prop_dcid)) is not None:
        return f" {wording.noun(axis).preposition} "
    return HOW_CONNECTIVES.get(prop_dcid, DEFAULT_HOW_CONNECTIVE)


def role_class_text(label: str, prop_dcid: str | None, wording: RoleWording) -> str:
    """Name a role-class value as that role's entities: the label, then the role's plural
    noun unless the label already ends in a plural ("bilateral" reads "bilateral
    creditors", "bondholders" stays). Matching the final "s" is a text rule. The ten IDS
    creditor-type labels it reads are pinned in tests/render/test_render_heading_words.py."""
    axis = ROLE_CLASS_PROPERTIES.get(prop_dcid) if prop_dcid is not None else None
    if axis is None or label.endswith("s"):
        return label
    return f"{label} {wording.noun(axis).plural}"


HOW_PHRASE_LEADS_WITH_PREPOSITION = frozenset(
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
"""The prepositions a value phrase can open with, which then needs no connective of its own
("with conditions" in place of "for with conditions")."""


def connective_for_phrase(prop_dcid: str | None, phrase: str, wording: RoleWording) -> str:
    """Return the connective for a value phrase after the noun, or a bare space when the
    phrase starts with a preposition. An adjectival property's value that follows the
    noun (an open axis's agreed value) reads `DEFAULT_HOW_CONNECTIVE`."""
    first_word = phrase.split(" ", 1)[0].lower()
    if first_word in HOW_PHRASE_LEADS_WITH_PREPOSITION:
        return " "
    return how_connective(prop_dcid, wording) or DEFAULT_HOW_CONNECTIVE


def how_value_text(
    dcid: str, label: str, *, prop: str | None, property_label: str | None, index: CodeIndex
) -> str:
    """A code's text after its property's connective (`HOW_CONNECTIVES`): its short name
    (`CodeRow.short_name`, "standard grants" for `CRSTypeOfFinance110`) when its code row
    belongs to `prop`, else `label` without the property suffix (`without_property_suffix`).
    Search's heading and the description composer's title read it."""
    row = index.code_by_dcid(dcid)
    if row is not None and row.short_name and row.code_system == prop:
        return row.short_name
    return without_property_suffix(label, property_label)


def without_property_suffix(label: str, property_label: str | None) -> str:
    """Drop a trailing bracketed property name the graph adds to tell code lists apart:
    "bilateral (IDS creditor type)" reads "bilateral"."""
    if not property_label:
        return label
    suffix = f" ({property_label})"
    return label[: -len(suffix)] if label.lower().endswith(suffix.lower()) else label


def is_price_basis(axis: str, value: str | None) -> bool:
    """Whether a measure axis value states a price basis (`PRICES_BY_QUALIFIER`)."""
    return axis == PRICE_BASIS_AXIS and value in PRICES_BY_QUALIFIER


# The identity `shape` member each `kind="shape"` measure axis reads.
_SHAPE_AXIS_FIELDS = {
    "populationType": "population_type",
    "measuredProperty": "measured_property",
    "measurementQualifier": "measurement_qualifier",
}


def normalize_label(dcid: str, raw_label: str, *, index: CodeIndex) -> str:
    """Return the index label for `dcid` when the index holds a row, otherwise `raw_label`.

    The graph names a code with its code-system suffix (`standard grant (CRS Finance)`). The
    index label drops it. Every surface that shows a graph node label passes it through here.
    """
    row = index.code_by_dcid(dcid)
    return row.label if row is not None else raw_label


def compose_measure_label(match: PartitionLabelMatch, values: Sequence[MeasureValueRow]) -> str:
    """The wire-facing label for one measure partition: `match.label` with each of
    `match.uncovered`'s `(axis, value)` pairs folded back in, except a price basis.

    An authored `MeasurePartitionLabel` omits any axis whose value it folds out of the
    rendered text (see `label_for_pairs`), so `match.label` alone can name two partitions
    identically. Each uncovered pair's `value` is looked up by dcid in `values` and appended
    parenthetically, in `uncovered`'s order, which keeps the published labels distinct.
    A price basis (`is_price_basis`) is skipped, because the answer's unit phrase states it
    and the nominal and real partitions read alike. Returns `match.label` unchanged when
    nothing else is uncovered. A `value` dcid absent from `values` is a measure index
    inconsistency and raises `KeyError`.
    """
    uncovered = [dcid for axis, dcid in match.uncovered if not is_price_basis(axis, dcid)]
    if not uncovered:
        return match.label
    by_dcid = {value.dcid: value for value in values}
    qualifiers = ", ".join(by_dcid[dcid].label for dcid in uncovered)
    return f"{match.label} ({qualifiers})"


def family_label(
    identity: Mapping[str, object], *, index: CodeIndex, measure_index: MeasureIndex
) -> str | None:
    """Name an identity's family: its measure partition label, then its bound slots' labels.

    The measure partition is read from the identity's shape and its value-bound measure-axis
    slots, and labelled with `compose_measure_label`. The other slots follow in slot order.
    `value` and `exact` bindings contribute their one ref, and a `set` binding joins its
    refs' labels with `/`. Measure-axis slots, slots pinned by the dataset the identity
    elects (`code_index.members.elected_pins`), unbound slots and refs the index holds no
    row for contribute nothing. A composite identity elects as its own shape and slots do,
    with no provenance pin (`code_index.composites.without_composite`), and is named by its
    declaration (`CompositeDeclaration.label`): after the measure label for a stacked
    composite, whose components share the measure, and in its place for a spliced one, whose
    measure changes between its periods (the OECD headline). Returns `None` when neither part
    has a label. The presentation door labels a collapsed family's `dcid:<spec_id>` ref with
    this, and the datasets door's `stat_vars_summed` warning carries the same string as `stat_var_label`.
    """
    shape = cast("Mapping[str, object]", identity.get("shape", {}))
    slots = cast("Sequence[Mapping[str, object]]", identity.get("slots", ()))
    bound = {cast(str, slot["property"]): _bound_refs(slot) for slot in slots}

    measure_axes = {axis.axis for axis in measure_index.axes}
    pairs = [
        (
            axis.axis,
            cast("str | None", shape.get(_SHAPE_AXIS_FIELDS[axis.axis]))
            if axis.kind == "shape"
            else _sole_ref(bound.get(axis.axis, ())),
        )
        for axis in measure_index.axes
    ]
    match = label_for_pairs(pairs, measure_index.partition_labels)
    parts = [compose_measure_label(match, measure_index.values)] if match is not None else []
    ref = composite_ref(identity)
    composite = index.composite(ref) if ref is not None else None
    if composite is not None:
        parts = [*(parts if composite.kind == "stacked" else []), composite.label]

    pinned = elected_pins(index, identity_family_query(index, without_composite(identity)))
    for prop, refs in bound.items():
        if prop in measure_axes or prop in pinned:
            continue
        slot_labels = tuple(
            row.label for ref in refs if (row := index.code_by_dcid(ref)) is not None
        )
        if slot_labels:
            parts.append("/".join(slot_labels))
    return ", ".join(parts) if parts else None


def index_family_label(
    family: IndexFamily,
    *,
    index: CodeIndex,
    measure_index: MeasureIndex,
    denominator_noun: str | None,
    population_type_label: str,
) -> str:
    """Name an index family by what every one of its rows shares, and its serving dataset.

    The label leads with `population_type_label` and the measure partition label of the
    family's shape axes and of the constraint measure axes every row holds at one value
    (`compose_measure_label`), as in "Financial flow disbursements", else with
    `population_type_label` alone. A ratio adds "as a share of" `denominator_noun`. The codes of
    the other properties every row holds at one value follow, by `CodeRow.label` in property
    order, except those the partition label's text names. The serving dataset's
    `DatasetDeclaration.label` closes the label. A property whose value differs across rows
    contributes nothing, so the label is the family's whichever member recall ranks first.
    Search labels a shape whose family the index decides with this.
    """
    shared = _shared_constraints(family)
    five_tuple = family.five_tuple
    pairs = [
        (
            axis.axis,
            cast("str | None", getattr(five_tuple, _SHAPE_AXIS_FIELDS[axis.axis]))
            if axis.kind == "shape"
            else shared.get(axis.axis),
        )
        for axis in measure_index.axes
    ]
    match = label_for_pairs(pairs, measure_index.partition_labels)
    measure = (
        f"{population_type_label} {_lead_lower(compose_measure_label(match, measure_index.values))}"
        if match is not None
        else population_type_label
    )
    if five_tuple.measurement_denominator is not None and denominator_noun is not None:
        measure = f"{measure} as a share of {denominator_noun}"

    measure_axes = {axis.axis for axis in measure_index.axes}
    named = frozenset(match.named_values) if match is not None else frozenset()
    parts = [measure]
    for prop, value in shared.items():
        if prop in measure_axes or (prop, value) in named:
            continue
        if (row := index.code_by_dcid(value)) is not None:
            parts.append(row.label)
    if family.datasets and (dataset := index.datasets.get(family.datasets[0])) is not None:
        parts.append(dataset.label)
    label = ", ".join(parts)
    return label[:1].upper() + label[1:]


def _lead_lower(text: str) -> str:
    """`text` with its first letter lowercased, unless its first word is an acronym."""
    return text[:1].lower() + text[1:] if text[1:2].islower() else text


def _shared_constraints(family: IndexFamily) -> dict[str, str]:
    """The constraint properties every row of `family` declares at one value, sorted."""
    first, *rest = family.rows
    return {
        prop: value
        for prop, value in sorted(first.constraint_properties.items())
        if all(row.constraint_properties.get(prop) == value for row in rest)
    }


def _bound_refs(slot: Mapping[str, object]) -> tuple[str, ...]:
    """The refs a slot's binding names: one for `value` and `exact`, all for `set`."""
    binding = cast("Mapping[str, object]", slot.get("binding", {}))
    state = binding.get("state")
    if state in ("value", "exact"):
        return (cast(str, binding["ref"]),)
    if state == "set":
        return tuple(cast("Sequence[str]", binding["refs"]))
    return ()


def _sole_ref(refs: tuple[str, ...]) -> str | None:
    return refs[0] if len(refs) == 1 else None
