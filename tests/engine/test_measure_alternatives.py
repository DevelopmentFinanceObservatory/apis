"""Tests measure-label composition and publication of sibling alternatives.

Synthetic partitions pin label rules, and five real fixture-index partitions check integration.
"""

from __future__ import annotations

import pytest

import search.engine.spec_build as spec_build_module
from code_index.labels import compose_measure_label
from code_index.shared import get_index, get_measure_index
from code_index.types import (
    MeasureAxis,
    MeasureIndex,
    MeasurePartitionLabel,
    MeasureValueRow,
    PartitionLabelMatch,
    label_for_pairs,
)
from search.engine.retrieval.bound_values import (
    MeasurePartition,
    _measure_axis_value,
    sibling_partitions,
)
from search.engine.retrieval.member_facts import IndexMemberFacts
from search.engine.shape import identity_shape_for
from search.engine.spec_build import (
    _measure_alternatives,
    measure_axis_values,
    measure_label_for_partition,
)
from search.models import (
    BindingValue,
    GraphRef,
    Slot,
    SlotKey,
    SlotValue,
    SpecAlternativeMeasure,
    SpecDefinition,
)
from tests.engine._base_identity import make_base
from tests.engine._builders import conjoin_spec, shape_five_tuple


def _value(dcid: str, axis: str, label: str) -> MeasureValueRow:
    return MeasureValueRow(
        dcid=dcid, axis=axis, label=label, synonyms=(), short_name=None, matchable=True
    )


_NETTING_VALUES = (_value("Gross", "nettingBasis", "gross"), _value("Net", "nettingBasis", "net"))


def test_a_fully_covering_match_is_returned_unchanged():
    match = PartitionLabelMatch(label="commitments", uncovered=())
    assert compose_measure_label(match, _NETTING_VALUES) == "commitments"


def test_uncovered_pairs_are_folded_in_by_their_own_readable_text():
    match = PartitionLabelMatch(label="disbursements", uncovered=(("nettingBasis", "Net"),))
    assert compose_measure_label(match, _NETTING_VALUES) == "disbursements (net)"


def test_two_partitions_an_authored_key_names_identically_compose_distinct_text():
    gross = PartitionLabelMatch(label="disbursements", uncovered=(("nettingBasis", "Gross"),))
    net = PartitionLabelMatch(label="disbursements", uncovered=(("nettingBasis", "Net"),))
    assert compose_measure_label(gross, _NETTING_VALUES) != compose_measure_label(
        net, _NETTING_VALUES
    )


def test_a_qualifier_that_is_no_price_basis_stays_in_the_label():
    match = PartitionLabelMatch(
        label="disbursements",
        uncovered=(("measurementQualifier", "Adjusted"), ("nettingBasis", "Net")),
    )
    values = (*_NETTING_VALUES, _value("Adjusted", "measurementQualifier", "adjusted"))
    assert compose_measure_label(match, values) == "disbursements (adjusted, net)"


@pytest.mark.parametrize("price_basis", ["Nominal", "RealValue"])
def test_the_price_basis_is_never_folded_into_a_label(price_basis):
    """The unit phrase states the price basis, so no label does, whatever its value."""
    match = PartitionLabelMatch(
        label="disbursements",
        uncovered=(("measurementQualifier", price_basis), ("nettingBasis", "Net")),
    )
    assert compose_measure_label(match, _NETTING_VALUES) == "disbursements (net)"


# Synthetic partitions cover a labelled and an unlabelled measuredProperty, a free
# populationType axis the label folds in, and a price basis the label leaves to the unit
# phrase.
_AXES = (
    MeasureAxis(axis="measuredProperty", kind="shape", role="basis", prefer=("count",)),
    MeasureAxis(axis="populationType", kind="shape", role="free", prefer=("Widget", "Gadget")),
    MeasureAxis(axis="measurementQualifier", kind="shape", role="free", prefer=("Nominal",)),
)

_LABELS = (
    # Covers count only. populationType is uncovered, and grams has no label.
    MeasurePartitionLabel(axis_values=(("measuredProperty", "count"),), label="count"),
)

_VALUES = (
    _value("count", "measuredProperty", "count"),
    _value("grams", "measuredProperty", "grams"),
    _value("Widget", "populationType", "widget"),
    _value("Gadget", "populationType", "gadget"),
    _value("Gizmo", "populationType", "gizmo"),
    _value("Nominal", "measurementQualifier", "nominal"),
)


def _elected_shape(meas_prop: str, pop_type: str) -> dict:
    return identity_shape_for(
        pop_type_dcid=pop_type,
        meas_prop_dcid=meas_prop,
        stat_type_dcid="measuredValue",
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
    )


def _slot(prop: str, value: str) -> Slot:
    return Slot(
        key=SlotKey(axis="what", property=GraphRef(ref=prop, label=prop), label=prop),
        binding=BindingValue(
            value=SlotValue(ref=GraphRef(ref=value, label=value), value_kind="enum_value")
        ),
    )


def _partition(meas_prop: str, pop_type: str) -> MeasurePartition:
    return MeasurePartition(
        values=(
            ("measuredProperty", meas_prop),
            ("populationType", pop_type),
            ("measurementQualifier", "Nominal"),
        ),
        members=(f"SV_{meas_prop}_{pop_type}",),
    )


def _synthetic_measure_index() -> MeasureIndex:
    return MeasureIndex(axes=_AXES, values=_VALUES, partition_labels=_LABELS)


def _elected_identity(meas_prop: str, pop_type: str):
    _, identity = make_base(_elected_shape(meas_prop, pop_type), [])
    return identity


def test_no_measure_family_publishes_none_not_an_empty_list():
    identity = _elected_identity("count", "Widget")

    alternatives, (label, label_axes, _) = _measure_alternatives(identity, None, ())

    assert alternatives is None
    assert label is None
    assert label_axes == []


def test_siblings_reach_the_output_in_stable_order_with_distinct_labels(monkeypatch):
    monkeypatch.setattr(spec_build_module, "get_measure_index", _synthetic_measure_index)
    identity = _elected_identity("count", "Widget")
    elected_partition = _partition("count", "Widget")
    # Deliberately not alphabetical/canonical order, to prove the output
    # follows the caller's own siblings order rather than re-sorting it.
    siblings = (_partition("count", "Gadget"), _partition("count", "Gizmo"))

    alternatives, (elected_label, elected_label_axes, _) = _measure_alternatives(
        identity, elected_partition, siblings
    )

    assert alternatives is not None
    assert [a.label for a in alternatives] == ["count (gadget)", "count (gizmo)"]
    assert [
        [(v.axis, v.value.ref, v.value.label) for v in a.measure_axis_values] for a in alternatives
    ] == [
        [
            ("measuredProperty", "count", "count"),
            ("populationType", pop_type, pop_type.lower()),
            ("measurementQualifier", "Nominal", "nominal"),
        ]
        for pop_type in ("Gadget", "Gizmo")
    ]
    assert elected_label == "count (widget)"
    # Every axis is non-None on elected_partition. measuredProperty folds into
    # "count", populationType folds in parenthetically, and the label leaves the
    # price basis to the unit phrase. All three are published as consumed, so no
    # how clause repeats any of them.
    assert set(elected_label_axes) == {"measuredProperty", "populationType", "measurementQualifier"}


def test_every_alternative_spec_id_differs_from_the_elected_and_from_each_other(monkeypatch):
    monkeypatch.setattr(spec_build_module, "get_measure_index", _synthetic_measure_index)
    elected_spec_id, identity = make_base(_elected_shape("count", "Widget"), [])
    elected_partition = _partition("count", "Widget")
    siblings = (_partition("count", "Gizmo"), _partition("count", "Gadget"))

    alternatives, _ = _measure_alternatives(identity, elected_partition, siblings)

    assert alternatives is not None
    spec_ids = [a.definition.spec_id for a in alternatives]
    assert elected_spec_id not in spec_ids
    assert len(set(spec_ids)) == len(spec_ids)


def test_a_sibling_matching_no_authored_label_is_absent_with_no_exception(monkeypatch):
    monkeypatch.setattr(spec_build_module, "get_measure_index", _synthetic_measure_index)
    identity = _elected_identity("count", "Widget")
    elected_partition = _partition("count", "Widget")
    # "grams" carries no authored label at all: an unlabelled sibling is
    # dropped silently, not raised as an error.
    siblings = (_partition("count", "Gizmo"), _partition("grams", "Widget"))

    alternatives, _ = _measure_alternatives(identity, elected_partition, siblings)

    assert alternatives is not None
    assert len(alternatives) == 1
    assert alternatives[0].label == "count (gizmo)"


def test_the_elected_label_is_composed_the_same_way_as_alternatives(monkeypatch):
    monkeypatch.setattr(spec_build_module, "get_measure_index", _synthetic_measure_index)
    identity = _elected_identity("count", "Gizmo")
    elected_partition = _partition("count", "Gizmo")

    _, (elected_label, _, _) = _measure_alternatives(identity, elected_partition, ())

    match = label_for_pairs(elected_partition.values, _LABELS)
    assert match is not None
    assert elected_label == compose_measure_label(match, _VALUES)


def test_measure_label_for_partition_agrees_with_measure_alternatives(monkeypatch):
    """`_measure_alternatives` and `measure_label_for_partition` return the same label."""
    monkeypatch.setattr(spec_build_module, "get_measure_index", _synthetic_measure_index)
    identity = _elected_identity("count", "Widget")
    elected_partition = _partition("count", "Widget")
    siblings = (_partition("count", "Gadget"), _partition("count", "Gizmo"))

    _, elected_label = _measure_alternatives(identity, elected_partition, siblings)

    assert measure_label_for_partition(elected_partition) == elected_label


def test_an_elected_partition_with_no_authored_label_publishes_none_alongside_alternatives(
    monkeypatch,
):
    monkeypatch.setattr(spec_build_module, "get_measure_index", _synthetic_measure_index)
    identity = _elected_identity("grams", "Widget")
    elected_partition = _partition("grams", "Widget")
    siblings = (_partition("count", "Widget"),)

    alternatives, (elected_label, elected_label_axes, _) = _measure_alternatives(
        identity, elected_partition, siblings
    )

    assert elected_label is None
    assert elected_label_axes == []
    assert alternatives is not None
    assert alternatives[0].label == "count (widget)"


def test_alternative_measures_and_measure_label_never_affect_spec_id():
    shape = shape_five_tuple("Person", "count")
    spec = conjoin_spec(shape, "Val")

    fabricated = SpecAlternativeMeasure(
        definition=SpecDefinition(spec_id=spec.spec_id, identity=spec.definition.identity),
        label="a label",
        axis="what",
        measure_axis_values=[],
    )
    with_alternatives = spec.model_copy(
        update={
            "resolved": spec.resolved.model_copy(
                update={"alternative_measures": [fabricated], "measure_label": "elected label"}
            )
        }
    )

    assert with_alternatives.spec_id == spec.spec_id
    assert with_alternatives.definition == spec.definition


def test_real_index_every_partition_taken_as_elected_disambiguates_and_carries_a_label():
    index = get_index()
    axes = get_measure_index().axes
    axis_names = tuple(axis.axis for axis in axes)

    grouped: dict[tuple[str | None, ...], list[str]] = {}
    for row in index.statvars:
        key = tuple(_measure_axis_value(row, axis) for axis in axes)
        grouped.setdefault(key, []).append(row.dcid)
    partitions = tuple(
        MeasurePartition(values=tuple(zip(axis_names, key, strict=True)), members=tuple(dcids))
        for key, dcids in grouped.items()
    )
    assert len(partitions) == 9

    for elected_partition in partitions:
        siblings = sibling_partitions(elected_partition, partitions, axes)
        elected_values = dict(elected_partition.values)
        representative = next(
            row for row in index.statvars if row.dcid == elected_partition.members[0]
        )
        five_tuple = IndexMemberFacts(row=representative).read_five_tuple()
        elected_shape = identity_shape_for(
            pop_type_dcid=elected_values.get("populationType"),
            meas_prop_dcid=elected_values.get("measuredProperty"),
            stat_type_dcid=five_tuple.stat_type_dcid,
            meas_qual_dcid=elected_values.get("measurementQualifier"),
            meas_denom_dcid=five_tuple.meas_denom_dcid,
            meas_denom_unresolved=five_tuple.meas_denom_unresolved,
        )
        elected_slots = [
            _slot(axis.axis, value)
            for axis in axes
            if axis.kind == "constraint" and (value := elected_values.get(axis.axis)) is not None
        ]
        _, elected_identity = make_base(elected_shape, elected_slots)

        alternatives, (elected_label, elected_label_axes, _) = _measure_alternatives(
            elected_identity, elected_partition, siblings
        )

        assert alternatives is not None
        assert elected_label is not None
        all_labels = [elected_label, *(a.label for a in alternatives if a.axis == "what")]
        assert len(all_labels) == len(set(all_labels)), (
            f"colliding labels electing {elected_partition.values!r}: {all_labels!r}"
        )
        # Every axis this partition declares a value on is published as
        # consumed, the union compose_measure_label draws from, whether an
        # axis landed in the label's own text or its parenthetical.
        assert set(elected_label_axes) == {
            axis for axis, value in elected_partition.values if value is not None
        }
        # The measure index labels every value a fixture partition holds.
        assert [v.axis for v in measure_axis_values(elected_partition)] == elected_label_axes
        for alternative in alternatives:
            assert alternative.measure_axis_values


# A price basis is the unit token's option, and the measure menu names none. These
# partitions vary the qualifier alone.
_PRICED_AXES = (
    MeasureAxis(axis="measuredProperty", kind="shape", role="basis", prefer=("count",)),
    MeasureAxis(
        axis="measurementQualifier", kind="shape", role="free", prefer=("Nominal", "RealValue")
    ),
)
_PRICED_LABELS = (
    MeasurePartitionLabel(axis_values=(("measuredProperty", "count"),), label="count"),
    MeasurePartitionLabel(axis_values=(("measuredProperty", "mass"),), label="mass"),
)
_PRICED_VALUES = (
    _value("count", "measuredProperty", "count"),
    _value("mass", "measuredProperty", "mass"),
    _value("Nominal", "measurementQualifier", "nominal"),
    _value("RealValue", "measurementQualifier", "real terms"),
    _value("Adjusted", "measurementQualifier", "adjusted"),
)


def _priced_index() -> MeasureIndex:
    return MeasureIndex(axes=_PRICED_AXES, values=_PRICED_VALUES, partition_labels=_PRICED_LABELS)


def _priced_partition(meas_prop: str, qualifier: str) -> MeasurePartition:
    return MeasurePartition(
        values=(("measuredProperty", meas_prop), ("measurementQualifier", qualifier)),
        members=(f"SV_{meas_prop}_{qualifier}",),
    )


def _priced_identity(meas_prop: str, qualifier: str):
    _, identity = make_base(
        identity_shape_for(
            pop_type_dcid="Widget",
            meas_prop_dcid=meas_prop,
            stat_type_dcid="measuredValue",
            meas_qual_dcid=qualifier,
            meas_denom_dcid=None,
        ),
        [],
    )
    return identity


def test_a_price_basis_sibling_is_a_unit_option_that_switches_the_qualifier(monkeypatch):
    monkeypatch.setattr(spec_build_module, "get_measure_index", _priced_index)

    alternatives, (elected_label, _, _) = _measure_alternatives(
        _priced_identity("count", "RealValue"),
        _priced_partition("count", "RealValue"),
        (_priced_partition("count", "Nominal"),),
    )

    assert elected_label == "count"
    assert alternatives is not None
    (option,) = alternatives
    assert (option.axis, option.label) == ("unit", "current prices")
    assert option.definition.identity.shape.measurement_qualifier == "Nominal"


def test_the_measure_menu_offers_each_measure_once_at_the_elected_price_basis(monkeypatch):
    """Two siblings that differ only in price basis are one measure option."""
    monkeypatch.setattr(spec_build_module, "get_measure_index", _priced_index)
    siblings = (_priced_partition("mass", "RealValue"), _priced_partition("mass", "Nominal"))

    alternatives, _ = _measure_alternatives(
        _priced_identity("count", "Nominal"), _priced_partition("count", "Nominal"), siblings
    )

    assert alternatives is not None
    (option,) = alternatives
    assert (option.axis, option.label) == ("what", "mass")
    assert option.definition.identity.shape.measurement_qualifier == "Nominal"


def test_a_measure_held_only_at_another_price_basis_stays_on_the_measure_menu(monkeypatch):
    monkeypatch.setattr(spec_build_module, "get_measure_index", _priced_index)

    alternatives, _ = _measure_alternatives(
        _priced_identity("count", "Nominal"),
        _priced_partition("count", "Nominal"),
        (_priced_partition("mass", "RealValue"),),
    )

    assert alternatives is not None
    (option,) = alternatives
    assert (option.axis, option.label) == ("what", "mass")
    assert option.definition.identity.shape.measurement_qualifier == "RealValue"


def test_a_non_price_basis_qualifier_stays_in_the_label(monkeypatch):
    monkeypatch.setattr(spec_build_module, "get_measure_index", _priced_index)
    label, axes, _ = measure_label_for_partition(_priced_partition("count", "Adjusted"))
    assert label == "count (adjusted)"
    assert set(axes) == {"measuredProperty", "measurementQualifier"}
