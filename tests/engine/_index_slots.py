"""Slot keys of the fixture index's five-tuples, for test shapes and identities.

`derive_shapes` gives an index-carried shape one slot key per constraint property its
family's rows carry (`code_index.query.IndexFamily`), and the identity minted from it one
slot per key. The code index decides the membership of a query only when its slot keys cover
a row of one family, so a test shape or identity that reads an index population carries the
same keys. These helpers read a five-tuple's first family, the CRS one where the CRS
publishes it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from code_index import FamilyQuery
from code_index.query import CodeIndex
from code_index.shared import get_index, get_measure_index
from code_index.types import MeasureIndex
from search.engine.bind import SlotBindingDraft
from search.engine.qualifier_match import (
    CodedQualifier,
    QualifierSpan,
    TextReading,
    read_by_vocabulary,
)
from search.engine.region.phase_rank_bind import (
    _match_query_qualifiers,
    _QueryQualifiers,
    _read_sources,
)
from search.engine.region.state import shape_vocabulary
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


def crs_vocabulary() -> frozenset[str]:
    """The vocabulary of a CRS shape, as qualifier matching reads it: every property a
    family the CRS publishes carries."""
    return frozenset(
        prop
        for families in get_index().families_by_five_tuple.values()
        for family in families
        if "provenance/OECDCreditorReportingSystem" in family.datasets
        for prop in family.properties
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


def query_qualifiers(
    variable: str,
    dropped_qualifier_phrases: Sequence[str] = (),
    *,
    query_text: str | None = None,
    qualifiers: Sequence[CodedQualifier] | None = None,
    vocabularies: Iterable[frozenset[str]],
) -> _QueryQualifiers:
    """The query's matches, ties and owned phrases in each of `vocabularies`, as the prune
    step reads them from one reading of each text source."""
    vocabularies = tuple(vocabularies)
    sources = _read_sources(
        variable, dropped_qualifier_phrases, query_text=query_text, vocabularies=vocabularies
    )
    return _match_query_qualifiers(sources, qualifiers=qualifiers, vocabularies=vocabularies)


def read_text(
    text: str,
    index: CodeIndex,
    measure_index: MeasureIndex | None = None,
    *,
    vocabularies: Iterable[frozenset[str]] = (),
) -> TextReading:
    """`text` read in every code system and in each of `vocabularies`
    (`read_by_vocabulary`)."""
    (reading,) = read_by_vocabulary([text], index, measure_index, vocabularies=vocabularies)
    return reading


def own_variable_spans(
    variable: str, *shapes: ShapeDraft
) -> dict[frozenset[str], tuple[QualifierSpan, ...]]:
    """`variable`'s spans in each of `shapes`' vocabularies, as the prune step stores them."""
    reading = read_text(
        variable,
        get_index(),
        get_measure_index(),
        vocabularies=[shape_vocabulary(shape) for shape in shapes],
    )
    return {vocabulary: own.spans for vocabulary, own in reading.by_vocabulary.items()}
