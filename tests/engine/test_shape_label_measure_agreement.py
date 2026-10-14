"""Tests that a wire shape label describes the members a reading serves.

Labels combine the partition phrase with bound or shared axes. An unlabelled partition falls back
to shared values, then to a confirmed member's name. A reading with no partition takes the name
of the one member it serves.
"""

from __future__ import annotations

import pytest

from code_index.shared import get_index, get_measure_index
from search.engine.retrieval.bound_values import MeasurePartition, OpenAxisFact, _measure_axis_value
from search.engine.spec_build import build_spec
from search.models import (
    CoverageBare,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    Shape,
    SlotKey,
    StatVar,
)
from search.render.plan import render_sentence
from tests.engine._builders import conjoin_spec, gref, shape_five_tuple, value_slot

# Two real, packaged CRS StatVarRow dcids sharing every five-tuple/constraint
# value except nettingBasis, the same group derive_shapes would fold into
# one shape (grouping is by five-tuple alone; nettingBasis is a constraint
# property, not part of it).
_GROSS_DCID = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode11330_Nominal"
)
_NET_DCID = (
    "Amount_FinancialFlow_Net_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode11330_Nominal"
)
# The kind of raw catalog text a real ->name arc carries for these two rows:
# realistic in shape, not fetched live. What matters for these tests is that
# the two differ only in the gross/net word, and that neither string is a
# substring of the other.
_GROSS_CATALOG_NAME = (
    "Amount of financial flow (nominal): CRS purpose code 11330, "
    "CRS type of finance 110, disbursement, gross"
)
_NET_CATALOG_NAME = (
    "Amount of financial flow (nominal): CRS purpose code 11330, "
    "CRS type of finance 110, disbursement, net"
)


def _real_partition(dcid: str) -> MeasurePartition:
    index = get_index()
    axes = get_measure_index().axes
    row = next(r for r in index.statvars if r.dcid == dcid)
    return MeasurePartition(
        values=tuple((axis.axis, _measure_axis_value(row, axis)) for axis in axes),
        members=(dcid,),
    )


def _fixture_cross_check() -> None:
    """Both dcids really are packaged today, on opposite sides of
    nettingBasis. If a corpus refresh ever drops or reshapes either, the
    test below would otherwise pass or fail for the wrong reason."""
    by_dcid = {r.dcid: r for r in get_index().statvars}
    assert by_dcid[_GROSS_DCID].constraint_properties["nettingBasis"] == "Gross"
    assert by_dcid[_NET_DCID].constraint_properties["nettingBasis"] == "Net"


def test_fixture_cross_check():
    _fixture_cross_check()


def test_served_gross_rows_publish_a_composed_label_not_either_members_own_name():
    gross_partition = _real_partition(_GROSS_DCID)
    net_partition = _real_partition(_NET_DCID)

    # The label a Net sibling's own ->name arc would stamp on this
    # group, had it scored higher at recall: what shape.label carries
    # before build_spec composes its own label, regardless of which
    # partition the served rows occupy.
    shape = Shape(
        shape_id="FinancialFlow/amount/Nominal/",
        label=_NET_CATALOG_NAME,
        population_type=gref("FinancialFlow"),
        measured_property=gref("amount"),
        stat_type=gref("measuredValue"),
        measurement_qualifier=gref("Nominal"),
        slot_keys=[
            SlotKey(
                axis="how", property=gref("nettingBasis", "netting basis"), label="netting basis"
            )
        ],
        member_count=2,
    )
    netting_slot = value_slot("how", "nettingBasis", "netting basis", "Gross", "gross")

    spec = build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=shape,
        slots=[netting_slot],
        stat_vars=[
            StatVar(
                ref=gref(_GROSS_DCID, _GROSS_CATALOG_NAME),
                shape_id=shape.shape_id,
                slot_values=[],
                constraint_properties=[],
            )
        ],
        entities=[],
        coverage=CoverageBare(has_data=True),
        measure_partition=gross_partition,
        measure_siblings=(net_partition,),
    )

    # shape.label is composed here from measure_label, not from either
    # member's own catalog name. nettingBasis is already folded into
    # measure_label's own text (it is one of measure_label_axes for this
    # partition, see the fixture check above), so the bound netting_slot
    # is not re-appended as a qualifier and the two strings agree exactly.
    assert spec.resolved.shape.label == "gross disbursements"
    assert spec.resolved.shape.label != _GROSS_CATALOG_NAME
    assert spec.resolved.shape.label != _NET_CATALOG_NAME
    assert spec.resolved.measure_label == "gross disbursements"

    sentence = render_sentence(spec)
    assert "gross" in sentence.lower()
    assert "net " not in sentence.lower()
    assert "net." not in sentence.lower()


def test_a_shape_with_no_measure_family_keeps_its_own_label_object_unchanged():
    """Without a measure partition, `build_spec` preserves the original `Shape` object."""
    shape = shape_five_tuple("Person", "count")
    spec = conjoin_spec(shape, "Val")

    assert spec.resolved.shape is shape
    assert spec.resolved.shape.label == "Test"


_TOTAL = ("SV_total", "Outstanding external debt")
_SHORT_TERM = ("SV_short_term", "Outstanding short-term external debt")


@pytest.mark.parametrize(
    ("served", "expected"),
    [
        pytest.param([_TOTAL], _TOTAL[1], id="one-named-member-names-the-reading"),
        pytest.param([("SV_total", None)], _SHORT_TERM[1], id="one-unnamed-member"),
        pytest.param([_TOTAL, _SHORT_TERM], _SHORT_TERM[1], id="several-members"),
    ],
)
def test_a_reading_with_no_partition_is_labelled_by_the_one_series_it_serves(
    served: list[tuple[str, str | None]], expected: str
):
    """A shape arrives named after a sibling recall ranked first. The wire label is the
    served member's graph name when the reading serves one named member, else the shape's."""
    shape = Shape(
        shape_id="FinancialPosition/amount/Nominal/",
        label=_SHORT_TERM[1],
        population_type=gref("FinancialPosition"),
        measured_property=gref("amount"),
        stat_type=gref("measuredValue"),
        measurement_qualifier=gref("Nominal"),
        slot_keys=[],
        member_count=2,
    )

    spec = build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=shape,
        slots=[],
        stat_vars=[
            StatVar(
                ref=gref(dcid, name),
                shape_id=shape.shape_id,
                slot_values=[],
                constraint_properties=[],
            )
            for dcid, name in served
        ],
        entities=[],
        coverage=CoverageBare(has_data=True),
    )

    assert spec.resolved.measure_label is None
    assert spec.resolved.shape.label == expected


def test_an_unlabelled_partition_composes_a_label_from_what_the_group_shares():
    """Without a measure label, the bound slot value supplies the wire label."""
    shape = shape_five_tuple("Widget", "grams")
    served_partition = MeasurePartition(
        values=(("measuredProperty", "grams"),), members=("SV_grams",)
    )
    spec = conjoin_spec(shape, "Val", sv_dcid="SV_grams", measure_partition=served_partition)

    assert spec.resolved.measure_label is None
    assert spec.resolved.shape.label == "Val"
    assert spec.resolved.shape.label != "Test"


def test_an_unlabelled_partition_with_nothing_to_compose_falls_back_to_a_served_member_name():
    """When the served group shares no label values, use a confirmed member's name."""
    shape = Shape(
        shape_id="Widget/grams/",
        label="Test",
        population_type=gref("Widget"),
        measured_property=gref("grams"),
        stat_type=gref("measuredValue"),
        slot_keys=[],
        member_count=1,
    )
    served_partition = MeasurePartition(
        values=(("measuredProperty", "grams"),), members=("SV_grams",)
    )
    spec = build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=shape,
        slots=[],
        stat_vars=[
            StatVar(
                ref=gref("SV_grams", "grams (widget catalog name)"),
                shape_id=shape.shape_id,
                slot_values=[],
                constraint_properties=[],
            )
        ],
        entities=[],
        coverage=CoverageBare(has_data=True),
        measure_partition=served_partition,
    )

    assert spec.resolved.measure_label is None
    assert spec.resolved.measure_axis_values == []
    assert spec.resolved.shape.label == "grams (widget catalog name)"


def test_a_group_agreeing_on_an_open_axis_names_it_but_never_a_varying_one():
    """Labels include bound and constant open axes, but omit axes whose served members vary.

    Synthetic axis names fall back to their dcids because the packaged index has no labels for them.
    """
    gross_partition = _real_partition(_GROSS_DCID)
    net_partition = _real_partition(_NET_DCID)
    shape = Shape(
        shape_id="FinancialFlow/amount/Nominal/",
        label=_NET_CATALOG_NAME,
        population_type=gref("FinancialFlow"),
        measured_property=gref("amount"),
        stat_type=gref("measuredValue"),
        measurement_qualifier=gref("Nominal"),
        slot_keys=[],
        member_count=2,
    )
    donor_slot = value_slot("what", "syntheticDonor", "donor", "syntheticDonorValue")

    spec = build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=shape,
        slots=[donor_slot],
        stat_vars=[
            StatVar(
                ref=gref(_GROSS_DCID, _GROSS_CATALOG_NAME),
                shape_id=shape.shape_id,
                slot_values=[],
                constraint_properties=[],
            )
        ],
        entities=[],
        coverage=CoverageBare(has_data=True),
        measure_partition=gross_partition,
        measure_siblings=(net_partition,),
        open_axis_facts=(
            OpenAxisFact(
                property="syntheticConstantAxis",
                member_values={"a": "syntheticConstantValue", "b": "syntheticConstantValue"},
            ),
            OpenAxisFact(
                property="syntheticVaryingAxis",
                member_values={"a": "syntheticValueOne", "b": "syntheticValueTwo"},
            ),
        ),
    )

    label = spec.resolved.shape.label
    assert label.startswith("gross disbursements (")
    assert "syntheticDonorValue" in label
    assert "syntheticConstantValue" in label
    assert "syntheticValueOne" not in label
    assert "syntheticValueTwo" not in label
    # Each axis carries its value, the declared default netting basis included.
    assert [(v.axis, v.value.ref) for v in spec.resolved.measure_axis_values] == [
        (axis, value) for axis, value in gross_partition.values if value is not None
    ]
    assert ("nettingBasis", "Gross") in [
        (v.axis, v.value.ref) for v in spec.resolved.measure_axis_values
    ]
    assert all(v.value.label for v in spec.resolved.measure_axis_values)
    (alternative,) = spec.resolved.alternative_measures or []
    assert ("nettingBasis", "Net") in [
        (v.axis, v.value.ref) for v in alternative.measure_axis_values
    ]


def test_a_unit_request_keeps_the_native_price_basis_in_the_axis_values():
    """The axis values name the served partition, so a posted unit keeps Nominal, and the
    label states no price basis either way."""
    shape = shape_five_tuple("FinancialFlow", "amount", "Nominal")
    unit = IdentityFacetValueRequest(
        request=IdentityUnitRequest(currency="EUR", prices="constant", base_year=2024)
    )

    spec = build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=shape,
        slots=[],
        stat_vars=[
            StatVar(
                ref=gref(_GROSS_DCID),
                shape_id=shape.shape_id,
                slot_values=[],
                constraint_properties=[],
            )
        ],
        entities=[],
        coverage=CoverageBare(has_data=True),
        measure_partition=_real_partition(_GROSS_DCID),
        facet_bindings={"unit": unit},
    )

    assert spec.resolved.measure_label == "gross disbursements"
    axes = [v.axis for v in spec.resolved.measure_axis_values]
    assert "nettingBasis" in axes
    assert "measurementQualifier" in axes


def test_relabelling_never_moves_spec_id():
    """Changing the display label leaves identity and `spec_id` unchanged."""
    shape = shape_five_tuple("FinancialFlow", "amount", "Nominal")
    gross_partition = _real_partition(_GROSS_DCID)
    net_partition = _real_partition(_NET_DCID)

    unlabelled = conjoin_spec(shape, "Val", sv_dcid=_GROSS_DCID)
    relabelled = conjoin_spec(
        shape,
        "Val",
        sv_dcid=_GROSS_DCID,
        measure_partition=gross_partition,
        measure_siblings=[net_partition],
    )

    assert relabelled.resolved.shape.label != unlabelled.resolved.shape.label
    assert relabelled.spec_id == unlabelled.spec_id
    assert relabelled.definition == unlabelled.definition


def test_a_served_member_name_never_moves_spec_id():
    """The served member's name on a reading with no partition is display text only."""
    shape = shape_five_tuple("FinancialPosition", "amount", "Nominal")

    def spec_serving(name: str | None):
        return build_spec(
            open_roles=(),
            measure_elected_properties=frozenset(),
            shape=shape,
            slots=[],
            stat_vars=[
                StatVar(
                    ref=gref("SV_total", name),
                    shape_id=shape.shape_id,
                    slot_values=[],
                    constraint_properties=[],
                )
            ],
            entities=[],
            coverage=CoverageBare(has_data=True),
        )

    unnamed = spec_serving(None)
    named = spec_serving("Outstanding external debt")

    assert named.resolved.shape.label != unnamed.resolved.shape.label
    assert named.spec_id == unnamed.spec_id
    assert named.definition == unnamed.definition
