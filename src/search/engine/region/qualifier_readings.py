"""Choose a span's code or measure reading by the shape it is read against.

A span such as "PPP" names a CRS purpose code and a price basis (`QualifierReadings`).
Text cannot choose between them, and a shape can: the reading the shape carries is the
one the query meant for it.
"""

from __future__ import annotations

from collections.abc import Sequence

from code_index.shared import get_index, get_measure_index
from search.engine.bind import SlotBindingDraft
from search.engine.bind_repair import _label_named_in_text
from search.engine.qualifier_match import (
    QualifierMatch,
    QualifierReadings,
    _normalize,
    match_qualifier_readings,
)
from search.engine.retrieval.bound_values import (
    _MEASURE_AXIS_DRAFT_FIELDS,
    index_carries_shape,
    shape_carries_measure_value,
)
from search.engine.shape import ShapeDraft


def shape_carries_measure_match(shape: ShapeDraft, match: QualifierMatch) -> bool:
    """Whether `shape` carries the measure value `match` names, in any spelling of its reading.

    Index rows decide for a shape the index carries. Otherwise the shape's own five-tuple
    field decides, and a constraint axis, which only index rows state, is not carried.
    """
    index = get_index()
    measure_index = get_measure_index()
    (axis,) = (axis for axis in measure_index.axes if axis.axis == match.code_system)
    reading = measure_index.reading_set(match.code_dcid)
    if index_carries_shape(shape, index):
        return any(shape_carries_measure_value(shape, index, axis, value) for value in reading)
    return axis.kind == "shape" and getattr(shape, _MEASURE_AXIS_DRAFT_FIELDS[axis.axis]) in reading


def bound_value_reads_measure(
    shape: ShapeDraft, match: QualifierMatch, bindings: Sequence[SlotBindingDraft]
) -> bool:
    """Whether a value bound on `shape` reads the measure value `match` names.

    A constraint outside the measure axes can state a measure reading: IMF WEO's
    `currencyConversionBasis = PurchasingPowerParityRate` ("purchasing power parity rate")
    is the PPP reading of a series whose measurementQualifier is `Nominal`. A bound value
    reads `match` when its graph label names, as whole words, the label of a value in
    `match`'s reading. Synonyms stay out because short ones such as "real" name other
    things inside a label ("real estate").
    """
    measure_index = get_measure_index()
    reading = measure_index.reading_set(match.code_dcid)
    texts = [
        row.label
        for row in measure_index.values
        if row.axis == match.code_system and row.dcid in reading
    ]
    value_labels = shape.value_labels or {}
    bound_labels = [
        label
        for binding in bindings
        if binding.kind in ("value", "set")
        for value_dcid in binding.value_dcids
        if (label := value_labels.get(value_dcid)) is not None
    ]
    return any(_label_named_in_text(text, label) for text in texts for label in bound_labels)


def reading_for_shape(readings: QualifierReadings, shape: ShapeDraft) -> QualifierMatch | None:
    """The one reading `shape` carries, or `None` when it carries both or neither.

    The code reading is carried when the shape declares its code system as a constraint
    property, the test `_admit_qualifier_matches_against_elected_shape` applies to an
    ordinary match. The measure reading is carried per `shape_carries_measure_match`.
    """
    declared = {slot.property_dcid for slot in shape.slot_keys if slot.property_dcid}
    carried = [
        match
        for match, carries in (
            (readings.code, readings.code.code_system in declared),
            (readings.measure, shape_carries_measure_match(shape, readings.measure)),
        )
        if carries
    ]
    return carried[0] if len(carried) == 1 else None


def match_query_readings(
    variable: str,
    dropped_qualifier_phrases: Sequence[str] = (),
    *,
    query_text: str | None = None,
) -> tuple[tuple[QualifierReadings, ...], frozenset[str]]:
    """Collect two-reading spans from `variable`, `query_text` and each dropped phrase.

    Sources are matched separately, as `_match_query_qualifiers` matches them. The second
    value holds the normalised phrases of `variable` and `query_text` readings, and the
    raw-text path owns dropped phrases. A span inside a dataset mention is never a reading,
    since `match_qualifier_readings` leaves the mention's words unread.
    """
    index = get_index()
    measure_index = get_measure_index()
    owned = match_qualifier_readings(variable, index, measure_index)
    if query_text:
        owned += match_qualifier_readings(query_text, index, measure_index)
    readings = owned
    for phrase in dropped_qualifier_phrases:
        readings += match_qualifier_readings(phrase, index, measure_index)
    return (
        tuple(dict.fromkeys(readings)),
        frozenset(_normalize(r.matched_text) for r in owned),
    )
