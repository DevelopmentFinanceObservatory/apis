"""Shared builders for tests/render: minimal Spec/Slot/Entity construction and
the round-trip assertion every rendered sentence must satisfy.

``_spec`` is the minimal valid Spec every test in the package builds from, and
``recorded_spec`` rebuilds one recorded on the wire;
``_what_slot``/``_how_slot``/`_where_slot``/``_source_slot``/``_where_place_slot``
and the entity/gender/residence-type builders cover the slot and entity shapes
its tests bind. ``_gender_slot``, ``_residence_type_slot``, and
``_unbound_source_slot`` live here rather than beside a single test file
because more than one behaviour file constructs specs against them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scripts.sentence_golden import spec_stating_facts
from search.engine.spec_build import _resolved_entities, _resolved_stat_vars
from search.engine.spec_id import build_spec_identity
from search.models import (
    Axis,
    BindingUnbound,
    BindingValue,
    Coverage,
    CoverageBare,
    Entity,
    EntityRoleDirectional,
    EntityRoleSubject,
    GraphRef,
    IdentityFacetBinding,
    OpenConstraintAxisFact,
    ProvenanceRef,
    RequestedWindow,
    ResolutionTrace,
    ResolvedFacet,
    ResolvedScope,
    ResolvedSpec,
    SentenceToken,
    Shape,
    Slot,
    SlotKey,
    SlotValue,
    Spec,
    SpecDefinition,
    SpecObservations,
    StatVar,
    TimeWindow,
)
from search.render.plan import render_sentence, render_sentence_plan


def assert_round_trip(plan: list[SentenceToken], rendered: str) -> None:
    assert "".join(t.text for t in plan) == rendered


def _rendered(spec: Spec) -> str:
    """Render a spec and assert the result matches its sentence plan."""
    rendered = render_sentence(spec)
    assert_round_trip(render_sentence_plan(spec), rendered)
    return rendered


def recorded_spec(raw: Mapping[str, Any]) -> Spec:
    """A Spec recorded on the wire, stating the sentence facts stored beside it
    (`sentence_facts`), as the engine stated them when it answered."""
    spec = Spec.model_validate(
        {key: value for key, value in raw.items() if key != "sentence_facts"}
    )
    return spec_stating_facts(spec, dict(raw))


def _ref(dcid: str, label: str) -> GraphRef:
    return GraphRef(dcid=dcid, label=label)


def _slot_key(axis: Axis, prop_dcid: str, prop_label: str, label: str) -> SlotKey:
    return SlotKey(axis=axis, property=_ref(prop_dcid, prop_label), label=label)


def _spec(
    *,
    slots: list[Slot] | None = None,
    entities: list[Entity] | None = None,
    window: TimeWindow | None = None,
    measured_property_label: str = "Development Finance Flow",
    measured_property: GraphRef | None = None,
    population_type: GraphRef | None = None,
    stat_type: GraphRef | None = None,
    measurement_qualifier: GraphRef | None = None,
    measurement_denominator: GraphRef | None = None,
    stat_vars: list[StatVar] | None = None,
    resolved_sources: list[GraphRef] | None = None,
    observations: SpecObservations | None = None,
    coverage: Coverage | None = None,
    scope: ResolvedScope | None = None,
    facets: dict[str, ResolvedFacet] | None = None,
    requested_window: RequestedWindow | None = None,
    available_at_place: bool | None = None,
    open_axis_facts: list[OpenConstraintAxisFact] | None = None,
    measure_label: str | None = None,
    measure_label_axes: list[str] | None = None,
    identity_facets: dict[str, IdentityFacetBinding] | None = None,
) -> Spec:
    """Build a minimal valid Spec for sentence-rendering tests.

    Optional arguments set resolved fields, slots, coverage, identity window and facets,
    and labels.
    """
    shape = Shape(
        shape_id="test-shape",
        label="Test Shape",
        population_type=(
            population_type
            if population_type is not None
            else _ref("DevelopmentFinance", "Development Finance")
        ),
        measured_property=(
            measured_property
            if measured_property is not None
            else _ref("DevelopmentFinanceFlow", measured_property_label)
        ),
        stat_type=stat_type if stat_type is not None else _ref("measuredValue", "Measured Value"),
        measurement_qualifier=measurement_qualifier,
        measurement_denominator=measurement_denominator,
        slot_keys=[],
        member_count=1,
    )
    coverage = coverage if coverage is not None else CoverageBare(has_data=True, window=window)
    trace = ResolutionTrace(
        resolved_entities=[],
        resolved_sources=[
            ProvenanceRef(dcid=ref.dcid, label=ref.label) for ref in resolved_sources or []
        ],
        slot_filters=[],
    )
    # test-spec-id is a readable placeholder; render tests exercise only the
    # rendered text. definition must still be a genuine, hashable projection of
    # the SAME slots/entities the Spec carries, mirroring spec_build.build_spec's
    # own keystone (definition and spec_id minted from one projection).
    identity_shape = {"kind": "statvar_schema", "population_type": "test-shape"}
    _, identity_doc = build_spec_identity(
        identity_shape,
        slots or [],
        entities=entities,
        requested_window=requested_window,
        explicit_facets=identity_facets,
        open_roles=(),
    )
    members = (
        stat_vars
        if stat_vars is not None
        else [
            StatVar(
                ref=_ref("sv1", "SV1"),
                shape_id="test-shape",
                slot_values=[],
                constraint_properties=[],
            )
        ]
    )
    return Spec(
        spec_id="test-spec-id",
        definition=SpecDefinition(spec_id="test-spec-id", identity=identity_doc),
        resolved=ResolvedSpec(
            shape=shape,
            slots=slots or [],
            stat_vars=_resolved_stat_vars(
                members, member_count=None, member_count_exact=True, stated=None
            ),
            entities=_resolved_entities(entities or []),
            coverage=coverage,
            scope=scope,
            facets=facets or {},
            trace=trace,
            observations=observations,
            available_at_place=available_at_place,
            open_axis_facts=open_axis_facts,
            measure_label=measure_label,
            measure_label_axes=measure_label_axes or [],
        ).stating_members(members),
    )


def _what_slot(label: str) -> Slot:
    return Slot(
        key=_slot_key("what", "DevelopmentFinanceScheme", "Scheme", "scheme"),
        binding=BindingValue(value=SlotValue(ref=_ref("ODAGrants", label), value_kind="entity")),
    )


def _how_slot(label: str) -> Slot:
    return Slot(
        key=_slot_key("how", "DevelopmentFinancePurpose", "Purpose", "purpose"),
        binding=BindingValue(value=SlotValue(ref=_ref("DAC/Health", label), value_kind="entity")),
    )


def _where_slot(label: str) -> Slot:
    return Slot(
        key=_slot_key("where", "DevelopmentFinanceRecipient", "Recipient", "recipient"),
        binding=BindingValue(value=SlotValue(ref=_ref("country/ETH", label), value_kind="entity")),
    )


def _source_slot(dcid: str, label: str) -> Slot:
    return Slot(
        key=SlotKey(axis="source", property=None, label="source"),
        binding=BindingValue(value=SlotValue(ref=_ref(dcid, label), value_kind="source")),
    )


def _where_place_slot(dcid: str, label: str) -> Slot:
    """Build a where slot for a subject place without a constraint property."""
    return Slot(
        key=SlotKey(axis="where", property=None, label="place"),
        binding=BindingValue(value=SlotValue(ref=_ref(dcid, label), value_kind="entity")),
    )


def _entity_to(dcid: str, label: str) -> Entity:
    return Entity(
        ref=_ref(dcid, label),
        entity_type=None,
        role=EntityRoleDirectional(
            kind="directional",
            role=_ref("RecipientRole", "Recipient"),
            axis="destination",
        ),
    )


def _entity_from(dcid: str, label: str) -> Entity:
    return Entity(
        ref=_ref(dcid, label),
        entity_type=None,
        role=EntityRoleDirectional(
            kind="directional",
            role=_ref("DonorRole", "Donor"),
            axis="origin",
        ),
    )


def _entity_subject(dcid: str, label: str) -> Entity:
    return Entity(
        ref=_ref(dcid, label),
        entity_type=None,
        role=EntityRoleSubject(),
    )


def _gender_slot(dcid: str, label: str) -> Slot:
    return Slot(
        key=_slot_key("how", "gender", "Gender", "gender"),
        binding=BindingValue(value=SlotValue(ref=_ref(dcid, label), value_kind="entity")),
    )


def _residence_type_slot(dcid: str, label: str) -> Slot:
    return Slot(
        key=_slot_key("how", "residenceType", "Residence Type", "residence type"),
        binding=BindingValue(value=SlotValue(ref=_ref(dcid, label), value_kind="entity")),
    )


def _unbound_source_slot() -> Slot:
    return Slot(key=SlotKey(axis="source", property=None, label="source"), binding=BindingUnbound())


def _unbound_how_slot() -> Slot:
    """Build an unbound purpose slot for open-axis rendering tests."""
    return Slot(
        key=_slot_key("how", "DevelopmentFinancePurpose", "Purpose", "purpose"),
        binding=BindingUnbound(),
    )
