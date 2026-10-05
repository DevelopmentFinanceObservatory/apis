"""Build minimal engine specs and entities for offline assemble tests.
`_make_promoting_spec` leaves `what` unbound to exercise the SV-name fallback.
"""

from __future__ import annotations

from search.engine.bind import SlotBindingDraft
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.engine.spec_build import build_shape_model, build_slot, build_spec
from search.models import (
    CoverageBare,
    Diagnostics,
    Entity,
    EntityRoleDirectional,
    EntityRoleSubject,
    GraphRef,
    PipelineStep,
    QueryEcho,
    StatVar,
    Timing,
)


def _make_query_echo() -> QueryEcho:
    return QueryEcho(
        entry_path="raw_text",
        raw_query="test query",
        normalized_query="test query",
        extract_skipped=False,
    )


def _make_diagnostics() -> Diagnostics:
    return Diagnostics(
        engine_build="test",
        trace_id="test-trace",
        warnings=[],
        timing_ms=Timing(total=0),
    )


def _make_coverage(has_data: bool = True) -> CoverageBare:
    return CoverageBare(has_data=has_data)


def _make_pipeline_steps() -> list[PipelineStep]:
    return [PipelineStep(step="extract", ran=True)]


def _make_entity(dcid: str, label: str) -> Entity:
    return Entity(
        ref=GraphRef(dcid=dcid, label=label),
        entity_type=None,
        role=EntityRoleSubject(),
    )


_POP_TYPE_DCID = "TestPopulationType"
_MEAS_PROP_DCID = "testMeasuredProperty"
_STAT_TYPE_DCID = "measuredValue"


def _make_five_tuple_refs() -> dict[str, GraphRef]:
    return {
        _POP_TYPE_DCID: GraphRef(dcid=_POP_TYPE_DCID, label="Test Population Type"),
        _MEAS_PROP_DCID: GraphRef(dcid=_MEAS_PROP_DCID, label="Test Measured Property"),
        _STAT_TYPE_DCID: GraphRef(dcid=_STAT_TYPE_DCID, label="Measured Value"),
    }


def _standard_shape_draft() -> ShapeDraft:
    """A minimal three-slot (what/how/where) standard ShapeDraft, built
    directly rather than through any resolver. The fixtures below only need
    a shape shaped enough for build_shape_model/build_spec to ground."""
    return ShapeDraft(
        shape_id="test_shape",
        label="test shape",
        pop_type_dcid=_POP_TYPE_DCID,
        meas_prop_dcid=_MEAS_PROP_DCID,
        stat_type_dcid=_STAT_TYPE_DCID,
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(
            SlotKeyDraft(axis="what", property_dcid="TestSchemeProp", label="scheme"),
            SlotKeyDraft(axis="how", property_dcid="TestPurposeProp", label="purpose"),
            SlotKeyDraft(axis="where", property_dcid="TestPlaceProp", label="place"),
        ),
    )


def _make_spec(
    sv_dcid: str,
    purpose_dcid: str,
    place_dcid: str,
    *,
    member_count: int = 1,
    entities: list[Entity] | None = None,
    identity_entities: list[Entity] | None = None,
):
    """Build a standard three-slot Spec; default identity hashes one Ethiopia entity.

    Pass `identity_entities` when its identity differs from `entities`.
    """
    shape_draft = _standard_shape_draft()
    five_tuple_refs = _make_five_tuple_refs()

    # Build slots: scheme=ODAGrants (value), purpose=<purpose_dcid> (value),
    # place=<place_dcid> (value)
    scheme_draft = SlotBindingDraft(
        axis="what",
        property_dcid="TestSchemeProp",
        kind="value",
        value_dcids=["ODAGrants"],
    )
    purpose_draft = SlotBindingDraft(
        axis="how",
        property_dcid="TestPurposeProp",
        kind="value",
        value_dcids=[purpose_dcid],
    )
    place_draft = SlotBindingDraft(
        axis="where",
        property_dcid="TestPlaceProp",
        kind="value",
        value_dcids=[place_dcid],
    )

    scheme_ref = GraphRef(dcid="ODAGrants", label="ODA Grants")
    purpose_ref = GraphRef(dcid=purpose_dcid, label=purpose_dcid)
    place_ref = GraphRef(dcid=place_dcid, label=place_dcid)

    slot_key_models = []
    slots = []
    for slot_draft_key, binding_draft, grounded in [
        (shape_draft.slot_keys[0], scheme_draft, [scheme_ref]),
        (shape_draft.slot_keys[1], purpose_draft, [purpose_ref]),
        (shape_draft.slot_keys[2], place_draft, [place_ref]),
    ]:
        prop_dcid = slot_draft_key.property_dcid
        assert prop_dcid is not None
        prop_ref = GraphRef(dcid=prop_dcid, label=prop_dcid)
        slot = build_slot(slot_draft_key, binding_draft, grounded, property_ref=prop_ref)
        slots.append(slot)
        slot_key_models.append(slot.key)

    shape_model = build_shape_model(
        shape_draft,
        slot_key_models,
        five_tuple_refs,
        member_count=member_count,
        members=[sv_dcid],
    )

    sv_ref = GraphRef(dcid=sv_dcid, label=sv_dcid)
    stat_vars = [
        StatVar(ref=sv_ref, shape_id=shape_draft.shape_id, slot_values=[], constraint_properties=[])
    ]

    return build_spec(
        measure_elected_properties=frozenset(),
        shape=shape_model,
        slots=slots,
        stat_vars=stat_vars,
        entities=entities if entities is not None else [_make_entity("country/ETH", "Ethiopia")],
        coverage=_make_coverage(),
        pipeline_trace=_make_pipeline_steps(),
        identity_entities=identity_entities,
    )


# Its humanized name extends "test measured property", triggering the SV-name fallback.
_PROMOTING_SV_DCID = "Test_Measured_Property_GDP"


def _make_promoting_spec(purpose_dcid: str | None, place_dcid: str):
    """Leave `what` unbound so `_fallback_what` can promote to `_PROMOTING_SV_DCID`.

    `purpose_dcid=None` also leaves `how` unbound for candidate-merge tests.
    """
    shape_draft = _standard_shape_draft()
    five_tuple_refs = _make_five_tuple_refs()

    purpose_draft = (
        SlotBindingDraft(
            axis="how",
            property_dcid="TestPurposeProp",
            kind="value",
            value_dcids=[purpose_dcid],
        )
        if purpose_dcid is not None
        else None
    )
    place_draft = SlotBindingDraft(
        axis="where",
        property_dcid="TestPlaceProp",
        kind="value",
        value_dcids=[place_dcid],
    )
    purpose_grounded = [GraphRef(dcid=purpose_dcid, label=purpose_dcid)] if purpose_dcid else []
    place_ref = GraphRef(dcid=place_dcid, label=place_dcid)

    slot_key_models = []
    slots = []
    for slot_draft_key, binding_draft, grounded in [
        (shape_draft.slot_keys[0], None, []),
        (shape_draft.slot_keys[1], purpose_draft, purpose_grounded),
        (shape_draft.slot_keys[2], place_draft, [place_ref]),
    ]:
        prop_dcid = slot_draft_key.property_dcid
        assert prop_dcid is not None
        prop_ref = GraphRef(dcid=prop_dcid, label=prop_dcid)
        slot = build_slot(slot_draft_key, binding_draft, grounded, property_ref=prop_ref)
        slots.append(slot)
        slot_key_models.append(slot.key)

    shape_model = build_shape_model(
        shape_draft,
        slot_key_models,
        five_tuple_refs,
        member_count=1,
        members=[_PROMOTING_SV_DCID],
    )

    sv_ref = GraphRef(dcid=_PROMOTING_SV_DCID, label=_PROMOTING_SV_DCID)
    stat_vars = [
        StatVar(ref=sv_ref, shape_id=shape_draft.shape_id, slot_values=[], constraint_properties=[])
    ]

    return build_spec(
        measure_elected_properties=frozenset(),
        shape=shape_model,
        slots=slots,
        stat_vars=stat_vars,
        entities=[_make_entity("country/ETH", "Ethiopia")],
        coverage=_make_coverage(),
        pipeline_trace=_make_pipeline_steps(),
    )


def _directional_entity(dcid: str, axis: str, label: str | None = None) -> Entity:
    return Entity(
        ref=GraphRef(dcid=dcid, label=label or dcid),
        entity_type=None,
        role=EntityRoleDirectional(
            kind="directional",
            role=GraphRef(dcid="irrelevant", label="irrelevant"),
            axis=axis,
        ),
    )
