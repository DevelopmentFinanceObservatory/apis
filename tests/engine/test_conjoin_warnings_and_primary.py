"""Tests multislot residuals and coverage downgrade during same-shape merges."""

from __future__ import annotations

import dataclasses

from search.engine.conjoin import BINDING_FOLDED, combine_regions
from search.engine.envelope import now_ms
from search.engine.region.candidate_options import _elected_sibling_warning
from search.engine.region.codes import ELECTED_SHAPE_NO_DATA
from search.engine.same_shape import collapse_same_shape
from search.engine.spec_build import build_spec
from search.models import (
    BindingValue,
    CoverageBare,
    CoverageExact,
    PipelineStep,
    Shape,
    Slot,
    SlotKey,
    SlotValue,
    StatVar,
    TimeWindow,
)
from tests.engine._builders import constraint_value_slot as _value_slot
from tests.engine._builders import gref as _gr
from tests.engine._conjoin_fixtures import _PERSON_SHAPE, _region
from tests.engine.test_conjoin_measure_merge import _recipient_only_shape, _where_bound_spec

# ---------------------------------------------------------------------------
# collapse_same_shape — multislot residual rule
# ---------------------------------------------------------------------------


def test_same_shape_multislot_residual_warns():
    """Same five-tuple, >1 slot differs: keep first; residual carries the second's
    (variable_text, earliest_index)."""
    # Build two shapes with DIFFERENT constraint props so both slots differ
    shape_two_slots = Shape(
        shape_id="Person/count/",
        label="Test",
        population_type=_gr("Person"),
        measured_property=_gr("count"),
        stat_type=_gr("measuredValue"),
        measurement_qualifier=None,
        measurement_denominator=None,
        slot_keys=[
            SlotKey(axis="what", property=_gr("propA"), label="propA"),
            SlotKey(axis="how", property=_gr("propB"), label="propB"),
        ],
        member_count=1,
    )

    def _two_slot_spec(val_a: str, val_b: str):
        return build_spec(
            measure_elected_properties=frozenset(),
            shape=shape_two_slots,
            slots=[
                Slot(
                    key=SlotKey(axis="what", property=_gr("propA"), label="propA"),
                    binding=BindingValue(value=SlotValue(ref=_gr(val_a), value_kind="enum_value")),
                ),
                Slot(
                    key=SlotKey(axis="how", property=_gr("propB"), label="propB"),
                    binding=BindingValue(value=SlotValue(ref=_gr(val_b), value_kind="enum_value")),
                ),
            ],
            stat_vars=[],
            entities=[],
            coverage=CoverageBare(has_data=True),
            pipeline_trace=[PipelineStep(step="extract", ran=True)],
        )

    spec_1 = _two_slot_spec("A1", "B1")
    spec_2 = _two_slot_spec("A2", "B2")
    r1 = _region(spec_1, "var 1", earliest_index=0)
    r2 = _region(spec_2, "var 2", earliest_index=1)

    effective, residual = collapse_same_shape([r1, r2])

    # SAME_SHAPE_MULTISLOT: first stays effective, second is residual
    assert len(effective) == 1
    assert effective[0] is r1
    assert residual == [("var 2", 1)]


def test_combine_regions_keeps_multislot_refusal_parts_distinct():
    """A multi-slot refusal is not a folded response group."""
    shape_two_slots = Shape(
        shape_id="Person/count/",
        label="Test",
        population_type=_gr("Person"),
        measured_property=_gr("count"),
        stat_type=_gr("measuredValue"),
        measurement_qualifier=None,
        measurement_denominator=None,
        slot_keys=[
            SlotKey(axis="what", property=_gr("propA"), label="propA"),
            SlotKey(axis="how", property=_gr("propB"), label="propB"),
        ],
        member_count=1,
    )

    def _two_slot_spec(val_a: str, val_b: str):
        return build_spec(
            measure_elected_properties=frozenset(),
            shape=shape_two_slots,
            slots=[
                Slot(
                    key=SlotKey(axis="what", property=_gr("propA"), label="propA"),
                    binding=BindingValue(value=SlotValue(ref=_gr(val_a), value_kind="enum_value")),
                ),
                Slot(
                    key=SlotKey(axis="how", property=_gr("propB"), label="propB"),
                    binding=BindingValue(value=SlotValue(ref=_gr(val_b), value_kind="enum_value")),
                ),
            ],
            stat_vars=[
                StatVar(
                    ref=_gr(f"sv_{val_a}_{val_b}"),
                    shape_id=shape_two_slots.shape_id,
                    slot_values=[],
                    constraint_properties=[],
                )
            ],
            entities=[],
            coverage=CoverageBare(has_data=True),
            pipeline_trace=[PipelineStep(step="extract", ran=True)],
        )

    spec_a = _two_slot_spec("A1", "B1")
    spec_b = _two_slot_spec("A2", "B2")
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    response, effective = combine_regions(
        [r_a, r_b],
        query="var A and var B",
        extra_warnings=[],
        start_ms=now_ms(),
        engine_build="test",
        include_sentence=False,
        max_candidates=10,
    )

    assert [part.spec.spec_id for part in response.parts] == [spec_a.spec_id, spec_b.spec_id]
    assert all(
        BINDING_FOLDED not in [warning.code for warning in part.warnings] for part in response.parts
    )
    assert effective == [r_a, r_b]
    assert effective[0] is r_a
    assert effective[1] is r_b


# ---------------------------------------------------------------------------
# collapse_same_shape — CoverageExact downgrade
# ---------------------------------------------------------------------------


def test_collapse_same_shape_exact_coverage_downgrade_unions_has_data():
    """The pure merge cannot recount observations for unioned StatVars.
    It must drop the first member's exact count and retain `has_data`.
    """
    window = TimeWindow(start_year=2020, end_year=2020)
    spec_a = build_spec(
        measure_elected_properties=frozenset(),
        shape=_PERSON_SHAPE,
        slots=[_value_slot("constraintProp", "ValA")],
        stat_vars=[
            StatVar(
                ref=_gr("sv_a"),
                shape_id=_PERSON_SHAPE.shape_id,
                slot_values=[],
                constraint_properties=[],
            )
        ],
        entities=[],
        coverage=CoverageExact(has_data=False, observation_count=0, dimensions=None, window=window),
        pipeline_trace=[PipelineStep(step="extract", ran=True)],
    )
    spec_b = build_spec(
        measure_elected_properties=frozenset(),
        shape=_PERSON_SHAPE,
        slots=[_value_slot("constraintProp", "ValB")],
        stat_vars=[
            StatVar(
                ref=_gr("sv_b"),
                shape_id=_PERSON_SHAPE.shape_id,
                slot_values=[],
                constraint_properties=[],
            )
        ],
        entities=[],
        coverage=CoverageExact(has_data=True, observation_count=30, dimensions=None, window=window),
        pipeline_trace=[PipelineStep(step="extract", ran=True)],
    )
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    assert effective[0].spec.resolved.coverage.kind != "exact", (
        "merged spec must not claim member-0's exact observation_count for the unioned stat_vars"
    )
    assert effective[0].spec.resolved.coverage.has_data is True
    assert effective[0].spec.resolved.coverage.window == window


def test_a_merge_folds_each_parts_elected_shape_no_data_into_one_union():
    """Two parts re-elect different readings. The merged answer warns once and names
    every reading either part dropped and elected, in first-seen order."""
    shape = _recipient_only_shape()
    spec_kenya = _where_bound_spec(shape, "country/KEN", "sv_kenya")
    spec_ethiopia = _where_bound_spec(shape, "country/ETH", "sv_ethiopia")
    r_a = dataclasses.replace(
        _region(spec_kenya, "aid to Kenya", earliest_index=0),
        warnings=(
            _elected_sibling_warning(no_data=["net"], elected=["gross"]).model_copy(
                update={"spec_id": spec_kenya.spec_id}
            ),
        ),
    )
    r_b = dataclasses.replace(
        _region(spec_ethiopia, "aid to Ethiopia", earliest_index=1),
        warnings=(
            _elected_sibling_warning(no_data=["commitments"], elected=["gross"]).model_copy(
                update={"spec_id": spec_ethiopia.spec_id}
            ),
        ),
    )

    effective, _residual = collapse_same_shape([r_a, r_b])

    (warning,) = effective[0].warnings
    assert warning.code == ELECTED_SHAPE_NO_DATA
    assert warning.spec_id == effective[0].spec.spec_id
    assert warning.params == {"no_data": ["net", "commitments"], "elected": ["gross"]}
