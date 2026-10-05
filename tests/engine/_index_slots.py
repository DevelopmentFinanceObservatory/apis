"""Slot keys of the packaged index's five-tuples, for test shapes and identities.

`derive_shapes` gives an index-carried shape one slot key per constraint property its
five-tuple's rows carry, and the identity minted from it one slot per key. The code index
decides the membership of a query only when its slot keys cover a row, so a test shape or
identity that reads an index population carries the same keys.
"""

from __future__ import annotations

from collections.abc import Mapping

from code_index import FamilyQuery
from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft
from search.engine.retrieval.bound_values import shape_query
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.models import IdentityBindingUnbound, IdentitySlot


def index_slot_properties(
    *,
    population_type: str,
    measured_property: str,
    stat_type: str = "measuredValue",
    measurement_qualifier: str | None = None,
) -> list[str]:
    """Every constraint property the index rows of the five-tuple carry, sorted."""
    return sorted(
        {
            prop
            for row in get_index().statvars
            if row.population_type == population_type
            and row.measured_property == measured_property
            and row.stat_type == stat_type
            and row.measurement_qualifier == measurement_qualifier
            and row.measurement_denominator is None
            for prop in row.constraint_properties
        }
    )


def index_slot_keys(**five_tuple: str | None) -> tuple[SlotKeyDraft, ...]:
    """A `how` slot key for each of `index_slot_properties`."""
    return tuple(
        SlotKeyDraft(axis="how", property_dcid=prop, label=prop)
        for prop in index_slot_properties(**five_tuple)
    )


def unbound_index_slots(**five_tuple: str | None) -> list[IdentitySlot]:
    """An unbound identity slot for each of `index_slot_properties`."""
    return [
        IdentitySlot(property=prop, binding=IdentityBindingUnbound())
        for prop in index_slot_properties(**five_tuple)
    ]


def query_for(shape: ShapeDraft, bound_values: Mapping[str, list[str]]) -> FamilyQuery:
    """The `shape_query` of `shape` under bindings that bind each property in `bound_values`
    to its values."""
    return shape_query(
        shape,
        [
            SlotBindingDraft(
                axis="how",
                property_dcid=prop,
                kind="value" if len(values) == 1 else "set",
                value_dcids=list(values),
            )
            for prop, values in bound_values.items()
        ],
    )
