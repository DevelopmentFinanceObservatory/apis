"""Slot keys of the packaged index's five-tuples, for test shapes and identities.

`derive_shapes` gives an index-carried shape one slot key per constraint property its
family's rows carry (`code_index.query.IndexFamily`), and the identity minted from it one
slot per key. The code index decides the membership of a query only when its slot keys cover
a row of one family, so a test shape or identity that reads an index population carries the
same keys. These helpers read a five-tuple's first family, the CRS one where the CRS
publishes it.
"""

from __future__ import annotations

from collections.abc import Mapping

from code_index import FamilyQuery
from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft
from search.engine.retrieval.bound_values import shape_query
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.models import IdentityBindingUnbound, IdentitySlot
from statvar_members import FiveTuple


def index_slot_properties(
    *,
    population_type: str,
    measured_property: str,
    stat_type: str = "measuredValue",
    measurement_qualifier: str | None = None,
) -> list[str]:
    """Every constraint property the rows of the five-tuple's first family carry, sorted."""
    five_tuple = FiveTuple(
        population_type=population_type,
        measured_property=measured_property,
        stat_type=stat_type,
        measurement_qualifier=measurement_qualifier,
        measurement_denominator=None,
    )
    return sorted(get_index().families_by_five_tuple[five_tuple][0].properties)


def index_slot_keys(**five_tuple: str | None) -> tuple[SlotKeyDraft, ...]:
    """A `how` slot key for each of `index_slot_properties`."""
    return tuple(
        SlotKeyDraft(axis="how", property_dcid=prop, label=prop)
        for prop in index_slot_properties(**five_tuple)
    )


def crs_flow_slot_keys() -> tuple[SlotKeyDraft, ...]:
    """The slot keys of the CRS shape of nominal financial-flow amounts."""
    return index_slot_keys(
        population_type="FinancialFlow", measured_property="amount", measurement_qualifier="Nominal"
    )


def crs_shape_systems() -> tuple[frozenset[str], ...]:
    """The constraint properties of a CRS shape, one set, as qualifier matching reads the
    shapes in play: every property a family the CRS publishes carries."""
    return (
        frozenset(
            prop
            for families in get_index().families_by_five_tuple.values()
            for family in families
            if "provenance/OECDCreditorReportingSystem" in family.datasets
            for prop in family.properties
        ),
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
        (),
    )
