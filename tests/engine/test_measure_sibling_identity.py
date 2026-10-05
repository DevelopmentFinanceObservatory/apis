"""Tests for sibling identity minting from an elected identity. Three synthetic
shape axes and one constraint axis isolate the rules; the final test uses the
packaged index.
"""

from __future__ import annotations

import hashlib

import pytest
import rfc8785

from code_index import CodeIndex, IndexStamp, StatVarRow
from code_index.shared import get_index, get_measure_index
from code_index.types import MeasureAxis, MeasurePartitionLabel
from search.engine.retrieval import bound_values as bound_values_module
from search.engine.retrieval.bound_values import (
    MeasurePartition,
    _measure_axis_value,
    elect_default_partition,
    family_asserted_measure_axes,
    mint_sibling_identity,
)
from search.engine.retrieval.member_facts import IndexMemberFacts
from search.engine.shape import identity_shape_for
from search.models import (
    BindingValue,
    Entity,
    EntityRoleSubject,
    GraphRef,
    IdentityEntityGroup,
    Slot,
    SlotKey,
    SlotValue,
    TimeWindow,
)
from tests.engine._base_identity import make_base

_AXES = (
    MeasureAxis(axis="populationType", kind="shape", role="free", prefer=("Widget",)),
    MeasureAxis(axis="measuredProperty", kind="shape", role="basis", prefer=("count",)),
    MeasureAxis(
        axis="measurementQualifier", kind="shape", role="free", prefer=("Nominal", "RealValue")
    ),
    MeasureAxis(axis="material", kind="constraint", role="free", prefer=("Plastic", "Metal")),
)

_ELECTED_SHAPE = identity_shape_for(
    pop_type_dcid="Widget",
    meas_prop_dcid="count",
    stat_type_dcid="measuredValue",
    meas_qual_dcid="Nominal",
    meas_denom_dcid=None,
)
_ELECTED_ENTITIES = [
    Entity(ref=GraphRef(dcid="place/1", label="Place One"), role=EntityRoleSubject())
]
_ELECTED_WINDOW = TimeWindow(start_year=2020, end_year=2020)


def _slot(prop: str, value: str) -> Slot:
    return Slot(
        key=SlotKey(axis="what", property=GraphRef(dcid=prop, label=prop), label=prop),
        binding=BindingValue(
            value=SlotValue(ref=GraphRef(dcid=value, label=value), value_kind="enum_value")
        ),
    )


_NON_MEASURE_SLOT = _slot("color", "Red")


def _elected() -> tuple[str, object]:
    return make_base(
        _ELECTED_SHAPE,
        [_NON_MEASURE_SLOT, _slot("material", "Plastic")],
        entities=_ELECTED_ENTITIES,
        requested_window=_ELECTED_WINDOW,
    )


def _partition(**overrides: str | None) -> MeasurePartition:
    values = {
        "populationType": "Widget",
        "measuredProperty": "count",
        "measurementQualifier": "Nominal",
        "material": "Plastic",
    }
    values.update(overrides)
    return MeasurePartition(values=tuple(values.items()), members=("SV_sibling",))


def _index_with_sibling_row(**constraint_properties: str) -> CodeIndex:
    """A packaged `CodeIndex` carrying exactly one `StatVarRow`, dcid
    `SV_sibling` (`_partition`'s own lone member), declaring
    `constraint_properties`. `mint_sibling_identity`'s own internal
    `get_index()` call is monkeypatched to this, standing in for the
    packaged index a sibling's real members would be found in.
    """
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="test-hash",
        ),
        declarations=(),
        codes=(),
        statvars=[
            StatVarRow(
                dcid="SV_sibling",
                population_type="Widget",
                measured_property="count",
                stat_type="measuredValue",
                measurement_qualifier="Nominal",
                measurement_denominator=None,
                observation_properties=(),
                constraint_properties=constraint_properties,
            )
        ],
    )


def test_minted_spec_id_matches_the_served_documents_hash() -> None:
    _, elected = _elected()

    result = mint_sibling_identity(elected, _partition(material="Metal"), _AXES, ())

    canonical = rfc8785.dumps(result.identity.model_dump(exclude_none=True))
    digest = hashlib.sha256(canonical).hexdigest()
    assert result.spec_id == f"spec_{digest[:32]}"


def test_sibling_differing_only_on_a_constraint_axis_changes_exactly_that_slot() -> None:
    elected_spec_id, elected = _elected()

    result = mint_sibling_identity(elected, _partition(material="Metal"), _AXES, ())

    elected_slots = {s.property: s for s in elected.slots}
    sibling_slots = {s.property: s for s in result.identity.slots}
    assert set(sibling_slots) == set(elected_slots)
    assert sibling_slots["color"] == elected_slots["color"]
    assert sibling_slots["material"] != elected_slots["material"]
    assert sibling_slots["material"].binding.ref == "Metal"
    assert result.spec_id != elected_spec_id


def test_sibling_differing_on_a_shape_axis_rebuilds_the_shape_member() -> None:
    elected_spec_id, elected = _elected()

    result = mint_sibling_identity(elected, _partition(measurementQualifier="RealValue"), _AXES, ())

    assert result.identity.shape.measurement_qualifier == "RealValue"
    # Undeclared five-tuple fields remain from the elected shape.
    assert result.identity.shape.population_type == elected.shape.population_type
    assert result.identity.shape.stat_type == elected.shape.stat_type
    assert result.spec_id != elected_spec_id


def test_undeclarable_shape_axis_raises_instead_of_being_silently_dropped() -> None:
    _, elected = _elected()
    axes = (
        *_AXES,
        MeasureAxis(axis="unitOfMeasure", kind="shape", role="free", prefer=("Count",)),
    )

    with pytest.raises(KeyError):
        mint_sibling_identity(elected, _partition(unitOfMeasure="Count"), axes, ())


def test_axis_the_partition_itself_never_asserts_is_omitted_not_left_unbound() -> None:
    """An axis absent from this partition's values produces no slot. Sibling
    partitions cannot supply it; `material` stands in for either real axis case.
    """
    _, elected = _elected()

    result = mint_sibling_identity(elected, _partition(material=None), _AXES, ())

    sibling_slots = {s.property: s for s in result.identity.slots}
    assert "material" not in sibling_slots

    # An absent value produces the same identity as an unrequested axis.
    never_asked_spec_id, _ = make_base(
        _ELECTED_SHAPE,
        [_NON_MEASURE_SLOT],
        entities=_ELECTED_ENTITIES,
        requested_window=_ELECTED_WINDOW,
    )
    assert result.spec_id == never_asked_spec_id


def test_family_asserted_measure_axes_reads_only_the_one_partition_passed() -> None:
    """Each call reports only axes asserted by its own partition."""
    axes = (
        *_AXES,
        MeasureAxis(axis="finish", kind="constraint", role="free", prefer=("Matte",)),
    )

    assert family_asserted_measure_axes(axes, _partition(material="Metal")) == {"material"}
    assert family_asserted_measure_axes(axes, _partition(material=None)) == frozenset()


def test_non_measure_slots_entities_and_window_are_byte_identical(monkeypatch) -> None:
    """Existing non-measure slots, entities, and the requested window stay
    unchanged when sibling identity minting adds no declared axis.
    """
    _, elected = _elected()
    monkeypatch.setattr(
        bound_values_module, "get_index", lambda: _index_with_sibling_row(material="Metal")
    )

    result = mint_sibling_identity(elected, _partition(material="Metal"), _AXES, ())

    sibling_slots = {s.property: s for s in result.identity.slots}
    assert sibling_slots["color"] == next(s for s in elected.slots if s.property == "color")
    assert result.identity.entities == elected.entities
    assert result.identity.requested_window == elected.requested_window


def test_a_grouped_elected_identity_mints_a_grouped_sibling_with_the_ungrouped_spec_id() -> None:
    _, elected = _elected()
    group = IdentityEntityGroup(dcid=None, label="the place pair")
    grouped = elected.model_copy(update={"entity_groups": {"observationAbout": group}})

    result = mint_sibling_identity(grouped, _partition(material="Metal"), _AXES, ())
    ungrouped = mint_sibling_identity(elected, _partition(material="Metal"), _AXES, ())

    assert result.spec_id == ungrouped.spec_id
    assert result.identity.entity_groups == {"observationAbout": group}


def test_sibling_mints_no_slot_for_an_axis_its_own_members_declare_that_elected_never_named(
    monkeypatch,
) -> None:
    """An axis declared by `SV_sibling` but absent from the elected identity is
    omitted. Resubmit slot keys come from the posted identity, so member facts
    cannot add `gauge`; existing `color` and `material` slots remain intact.
    """
    _, elected = _elected()
    monkeypatch.setattr(
        bound_values_module,
        "get_index",
        lambda: _index_with_sibling_row(material="Metal", gauge="Thin"),
    )

    result = mint_sibling_identity(elected, _partition(material="Metal"), _AXES, ())

    sibling_slots = {s.property: s for s in result.identity.slots}
    assert sibling_slots["color"] == next(s for s in elected.slots if s.property == "color")
    assert sibling_slots["material"].binding.ref == "Metal"
    assert "gauge" not in sibling_slots


def test_two_siblings_never_mint_the_same_spec_id() -> None:
    _, elected = _elected()

    by_material = mint_sibling_identity(elected, _partition(material="Metal"), _AXES, ())
    by_qualifier = mint_sibling_identity(
        elected, _partition(measurementQualifier="RealValue"), _AXES, ()
    )

    assert by_material.spec_id != by_qualifier.spec_id


def test_unlabelled_sibling_reports_the_absence_not_a_composed_label() -> None:
    _, elected = _elected()
    unrelated_label = MeasurePartitionLabel(
        axis_values=(("material", "Rubber"),), label="rubber widget"
    )

    result = mint_sibling_identity(elected, _partition(material="Metal"), _AXES, (unrelated_label,))

    assert result.label is None


def test_real_index_elected_and_siblings_each_mint_a_distinct_id_and_carry_a_label() -> None:
    index = get_index()
    axes = get_measure_index().axes
    labels = get_measure_index().partition_labels
    axis_names = tuple(axis.axis for axis in axes)

    grouped: dict[tuple[str | None, ...], list[str]] = {}
    for row in index.statvars:
        key = tuple(_measure_axis_value(row, axis) for axis in axes)
        grouped.setdefault(key, []).append(row.dcid)
    partitions = tuple(
        MeasurePartition(values=tuple(zip(axis_names, key, strict=True)), members=tuple(dcids))
        for key, dcids in grouped.items()
    )

    election = elect_default_partition(partitions, axes)
    assert election is not None
    assert election.siblings

    elected_values = dict(election.elected.values)
    representative = next(row for row in index.statvars if row.dcid == election.elected.members[0])
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
    elected_spec_id, elected = make_base(elected_shape, elected_slots)

    minted = [
        mint_sibling_identity(elected, sibling, axes, labels) for sibling in election.siblings
    ]

    spec_ids = {elected_spec_id, *(m.spec_id for m in minted)}
    assert len(spec_ids) == 1 + len(minted)
    assert all(m.label is not None for m in minted)
