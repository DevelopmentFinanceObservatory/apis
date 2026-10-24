"""The bind menu as the Jev binder reads it: what each shape offers, indexed for questions.

`jev_menu` reads a `BindRequest` the way the bind prompt lists it: each shape's own fixed
values, tie groups and candidate menus (`bind._qualifier_menu`), its listed constraint
slots (`bind._listed_slots`) and the values it offers on each, and the union of offered
values per slot across the menu. The request builder and the assembler both index slot
values by position in that union (`asked_values`), so they read one list.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from code_index.shared import get_index, get_measure_index
from code_index.types import MeasureValueRow
from search.engine.bind import (
    _MAX_TAXONOMY_VALUES_SHOWN,
    SlotBindingDraft,
    _display_taxonomy_values,
    _listed_slots,
    _qualifier_menu,
    _QualifierMenu,
    stated_reading,
)
from search.engine.binders.base import BindRequest
from search.engine.qualifier_match import QualifierMatch
from search.engine.shape import ShapeDraft
from search.engine.stated_words_read import (
    bound_value_label,
    implied_values,
    restricting,
    served_series,
)
from search.models import Axis

type Prop = tuple[Axis, str]
"""A constraint slot: its axis and property dcid."""

# Slot values asked about per property, from the start of the union.
VALUE_CAP = 60

# The ShapeDraft field each shape-kind measure axis reads.
AXIS_FIELDS: dict[str, str] = {
    "populationType": "pop_type_dcid",
    "measuredProperty": "meas_prop_dcid",
    "measurementQualifier": "meas_qual_dcid",
}


@dataclass(frozen=True, slots=True)
class JevMenu:
    """What the bind menu offers, per shape and per slot.

    `menus` holds each shape's fixed values, tie groups and candidate menus, `listed` its
    listed slots in order, and `offered` the values it offers on each. `properties` maps
    every slot any shape lists to its label and the union of offered values, shape by
    shape in menu order. `labels` maps a value dcid to its label, and `refused` holds the
    indices into `BindRequest.refused` of the readings a pinned part refused.
    """

    request: BindRequest
    menus: tuple[_QualifierMenu, ...]
    listed: tuple[tuple[Prop, ...], ...]
    offered: tuple[dict[Prop, list[str]], ...]
    properties: dict[Prop, tuple[str, list[str]]]
    labels: dict[str, str]
    refused: tuple[int, ...]


def jev_menu(request: BindRequest) -> JevMenu:
    """The menu `request` holds, as the bind prompt lists it.

    A shape offers on each listed slot its displayed taxonomy values (at most
    `bind._MAX_TAXONOMY_VALUES_SHOWN`), then the codes its candidate menus hold on that
    slot.
    """
    shapes = request.shapes
    menus = tuple(
        _qualifier_menu(
            shape,
            request.admitted_qualifiers[i] if request.admitted_qualifiers else (),
            request.admitted_ties[i] if request.admitted_ties else (),
            request.withheld_measure_axes,
            request.candidate_menus[i] if request.candidate_menus else (),
        )
        for i, shape in enumerate(shapes)
    )
    listed: list[tuple[Prop, ...]] = []
    offered: list[dict[Prop, list[str]]] = []
    properties: dict[Prop, tuple[str, list[str]]] = {}
    labels: dict[str, str] = {}
    for shape, menu in zip(shapes, menus, strict=True):
        slots: list[Prop] = []
        own: dict[Prop, list[str]] = {}
        taxonomy = shape.slot_taxonomy or {}
        for slot in _listed_slots(shape, request.withheld_measure_axes):
            assert slot.property_dcid is not None  # `_listed_slots` keeps none without one
            prop: Prop = (slot.axis, slot.property_dcid)
            slots.append(prop)
            shown = _display_taxonomy_values(
                shape, taxonomy.get(f"{slot.axis}:{slot.property_dcid}", [])
            )
            values = list(shown[:_MAX_TAXONOMY_VALUES_SHOWN])
            for value in values:
                if (label := _value_label(shape, value)) is not None:
                    labels.setdefault(value, label)
            for _text, rows in menu.candidates:
                for row in rows:
                    if row.code_system == slot.property_dcid and row.dcid not in values:
                        values.append(row.dcid)
                        labels.setdefault(row.dcid, row.label)
            own[prop] = values
            _label, union = properties.setdefault(prop, (slot.label, []))
            union.extend(v for v in values if v not in union)
        listed.append(tuple(slots))
        offered.append(own)
    refused = tuple(i for i, r in enumerate(request.refused) if r.pinned_failure is not None)
    return JevMenu(request, menus, tuple(listed), tuple(offered), properties, labels, refused)


def _value_label(shape: ShapeDraft, dcid: str) -> str | None:
    if shape.slot_taxonomy_index_sourced:
        row = get_index().code_by_dcid(dcid)
        return row.label if row is not None else None
    label = (shape.value_labels or {}).get(dcid)
    return label if label and label != dcid else None


def asked_values(menu: JevMenu, prop: Prop) -> list[str]:
    """The values of `prop` a question asks about, each at its position in the union: the
    first `VALUE_CAP` the menu offers."""
    return menu.properties[prop][1][:VALUE_CAP]


def prop_id(prop: Prop) -> str:
    """The question-id form of a slot, ``axis:property``."""
    return f"{prop[0]}:{prop[1]}"


def describe(
    shape: ShapeDraft,
    fixed: Sequence[QualifierMatch] = (),
    *,
    bindings: Sequence[SlotBindingDraft] = (),
    definition: bool = False,
    served: Sequence[str] = (),
) -> str:
    """One shape as a question quotes it: its label, then what its five-tuple, fixed values
    and the values `bindings` restrict it to (`restricting`) state beyond the label, and the
    slots left that select parts of it. A bound value reads by
    `stated_words_read.bound_value_label`, a set's values joined by "or". With
    `definition`, it also quotes the values the measure's definition fixes on properties
    `bindings` leave unrestricted (`stated_words_read.implied_values`), as the stated-word
    questions and Gemini's stated-words prompt do, and so do both for the `served` series
    an answer serving several serves (`stated_words_read.served_series`)."""
    reading = stated_reading(shape)
    parts = [f"{reading.measured_property} of {reading.population_type}"]
    if reading.qualifier:
        parts.append(reading.qualifier)
    if reading.denominator:
        parts.append(f"divided by {reading.denominator}")
    bound = restricting(bindings)
    restricted = [m.label for m in fixed] + [
        " or ".join(bound_value_label(shape, prop, dcid) for dcid in values)
        for prop, values in bound.items()
    ]
    if restricted:
        parts.append("restricted to " + ", ".join(dict.fromkeys(restricted)))
    if definition and (implied := implied_values(shape, bindings)):
        parts.append(f"its definition fixes {implied}")
    if served:
        parts.append(f"served as each of these series: {served_series(served)}")
    slots = [
        s.label
        for s in shape.slot_keys
        if s.property_dcid and s.axis != "source" and s.property_dcid not in bound
    ]
    if slots:
        parts.append("parts selectable by " + ", ".join(dict.fromkeys(slots)))
    return f"'{reading.label}' ({'; '.join(parts)})"


def measure_values(shapes: Sequence[ShapeDraft]) -> list[MeasureValueRow]:
    """The measure-axis values the shapes' five-tuples hold on a shape axis, each as its
    canonical reading (`MeasureIndex.canonical_reading`), less each axis's default (the
    first of its ``prefer``), in the measure index's order."""
    index = get_measure_index()
    held = {
        index.canonical_reading(getattr(shape, AXIS_FIELDS[axis.axis]) or "")
        for shape in shapes
        for axis in index.axes
        if axis.axis in AXIS_FIELDS
    }
    defaults = {axis.prefer[0] for axis in index.axes if axis.prefer}
    return [row for row in index.values if row.dcid in held and row.dcid not in defaults]
