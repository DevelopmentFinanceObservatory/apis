"""Build shared graph refs, slots, shapes, and specs for engine tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from search.engine.bind import SlotBindingDraft
from search.engine.region.candidates import _Candidate, _materialise_alt_shapes, elect_alternate
from search.engine.region.place_context import PlaceContext
from search.engine.region.place_roles import PlaceRoles
from search.engine.retrieval.bound_values import MeasurePartition, RoleClassWalks
from search.engine.retrieval.retrieve import UnsupportedDraft
from search.engine.shape import ShapeDraft, entities_by_role
from search.engine.spec_build import build_spec
from search.models import (
    Axis,
    BindingSet,
    BindingUnbound,
    BindingValue,
    CoverageBare,
    EntityRoleKey,
    GraphRef,
    Shape,
    Slot,
    SlotKey,
    SlotValue,
    StatVar,
)


def gref(dcid: str, label: str | None = None) -> GraphRef:
    """A GraphRef, labeled with `dcid` itself when no `label` is given."""
    return GraphRef(dcid=dcid, label=label or dcid)


def value_slot(
    axis: Axis,
    prop_dcid: str,
    prop_label: str,
    val_dcid: str,
    val_label: str | None = None,
) -> Slot:
    """A Slot with a BindingValue and a graph-ref value."""
    return Slot(
        key=SlotKey(axis=axis, property=gref(prop_dcid, prop_label), label=prop_label),
        binding=BindingValue(
            value=SlotValue(ref=gref(val_dcid, val_label), value_kind="enum_value")
        ),
    )


def unbound_slot(axis: Axis, prop_dcid: str, prop_label: str) -> Slot:
    """A Slot with a BindingUnbound."""
    return Slot(
        key=SlotKey(axis=axis, property=gref(prop_dcid, prop_label), label=prop_label),
        binding=BindingUnbound(),
    )


def binding_set_slot(axis: Axis, prop_dcid: str, prop_label: str, val_dcids: list[str]) -> Slot:
    """A Slot with a BindingSet over self-labeled graph-ref values."""
    return Slot(
        key=SlotKey(axis=axis, property=gref(prop_dcid, prop_label), label=prop_label),
        binding=BindingSet(
            values=[SlotValue(ref=gref(v), value_kind="enum_value") for v in val_dcids]
        ),
    )


def prop_node(dcid: str, types: list[str]) -> dict:
    """A graph node dict for a property/class node, in FakeGraph's node-fixture shape."""
    return {
        "name": dcid,
        "types": types,
        "dcid": dcid,
        "provenanceId": "dc/base/HumanReadableStatVars",
    }


def sv_arcs(
    *, pop_type: str, meas_prop: str, stat_type: str, label: str, gender: str | None
) -> dict:
    """Arc dict for a StatisticalVariable node, optionally gender-constrained."""
    arcs: dict = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": pop_type}]},
        "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
        "statType": {"nodes": [{"dcid": stat_type}]},
    }
    if gender is None:
        arcs["constraintProperties"] = {"nodes": []}
    else:
        arcs["constraintProperties"] = {"nodes": [{"dcid": "gender"}]}
        arcs["gender"] = {"nodes": [{"dcid": gender}]}
    return arcs


def shape_five_tuple(
    pop: str,
    mp: str,
    mq: str | None = None,
    *,
    denom: str | None = None,
    denom_unresolved: bool = False,
) -> Shape:
    """Build a five-tuple Shape with one `what` constraint slot.

    `denom_unresolved` distinguishes dcid-bearing or dcid-less unresolved targets from
    absent and resolved denominators.
    """
    return Shape(
        shape_id=f"{pop}/{mp}/{mq or ''}/{denom or ''}{'?' if denom_unresolved else ''}",
        label="Test",
        population_type=gref(pop),
        measured_property=gref(mp),
        stat_type=gref("measuredValue"),
        measurement_qualifier=gref(mq) if mq else None,
        measurement_denominator=gref(denom) if denom else None,
        measurement_denominator_unresolved=denom_unresolved,
        slot_keys=[SlotKey(axis="what", property=gref("constraintProp"), label="constraintProp")],
        member_count=1,
    )


def constraint_value_slot(prop: str, val: str) -> Slot:
    """A "what"-axis Slot binding `prop` to `val`, both self-labeled."""
    return Slot(
        key=SlotKey(axis="what", property=gref(prop), label=prop),
        binding=BindingValue(value=SlotValue(ref=gref(val), value_kind="enum_value")),
    )


def conjoin_spec(
    shape: Shape,
    constraint_val: str,
    sv_dcid: str = "test_sv",
    slot_options: list | None = None,
    entity_options: list | None = None,
    entities: list | None = None,
    inapplicable_slots: list[str] | None = None,
    unresolved_slots: list[str] | None = None,
    equivalent_readings: list | None = None,
    text_recognized_properties: frozenset[str] | None = None,
    measure_elected_properties: frozenset[str] | None = None,
    measure_partition: MeasurePartition | None = None,
    measure_siblings: list[MeasurePartition] | None = None,
):
    """Build a definite Spec with one constraint value slot and one StatVar.

    Provenance sets and measure-family metadata pass through to `build_spec`.
    """
    slot = constraint_value_slot("constraintProp", constraint_val)
    return build_spec(
        open_roles=(),
        shape=shape,
        slots=[slot],
        stat_vars=[
            StatVar(
                ref=gref(sv_dcid), shape_id=shape.shape_id, slot_values=[], constraint_properties=[]
            )
        ],
        entities=entities if entities is not None else [],
        coverage=CoverageBare(has_data=True),
        slot_options=slot_options,
        entity_options=entity_options,
        inapplicable_slots=inapplicable_slots,
        unresolved_slots=unresolved_slots,
        equivalent_readings=equivalent_readings,
        text_recognized_properties=text_recognized_properties or frozenset(),
        measure_elected_properties=measure_elected_properties or frozenset(),
        measure_partition=measure_partition,
        measure_siblings=measure_siblings or (),
    )


def subject_place_roles(*dcids: str) -> PlaceRoles:
    """A subject-only shape's binding of `dcids`, as `bind_place_roles` builds it."""
    return PlaceRoles(
        destination_dcids=[],
        origin_dcids=[],
        resolved_entities={"observationAbout": list(dcids)} if dcids else {},
        roles=[(dcid, "observationAbout") for dcid in dcids],
        where_dcids=list(dcids),
        obs_destination_dcid=dcids[0] if dcids else None,
    )


def bound_place(
    *,
    roles: Mapping[str, EntityRoleKey],
    entities: Mapping[str, list[str]],
    destination_dcid: str | None,
    origin_dcids: Sequence[str] = (),
    probe_dcids: Sequence[str] | None = None,
) -> PlaceContext:
    """A place context bound to named places: `roles` on the wire's role keys, `entities` by
    the shape's role names, `destination_dcid` as the observation fallback place, and
    `probe_dcids` as the probe places (`None` probes the one bound place unexpanded)."""
    return PlaceContext(
        roles=PlaceRoles(
            destination_dcids=[],
            origin_dcids=list(origin_dcids),
            resolved_entities={role: list(dcids) for role, dcids in entities.items()},
            roles=list(roles.items()),
            where_dcids=[destination_dcid] if destination_dcid else [],
            obs_destination_dcid=destination_dcid,
        ),
        probe_dcids=None if probe_dcids is None else tuple(probe_dcids),
        mode="bound",
        scope_warning=None,
    )


def flow_place_roles(
    shape: ShapeDraft, *, destination: Sequence[str] = (), origin: Sequence[str] = ()
) -> PlaceRoles:
    """A role-bearing shape's binding of places on each flow axis, as `bind_place_roles`
    builds it."""
    destination_dcids, origin_dcids = list(destination), list(origin)
    return PlaceRoles(
        destination_dcids=destination_dcids,
        origin_dcids=origin_dcids,
        resolved_entities=entities_by_role(
            shape=shape, by_axis={"destination": destination_dcids, "origin": origin_dcids}
        ),
        roles=[(dcid, "destination") for dcid in destination_dcids]
        + [(dcid, "origin") for dcid in origin_dcids],
        where_dcids=destination_dcids,
        obs_destination_dcid=destination_dcids[0] if destination_dcids else None,
    )


def bound_place_context(roles: PlaceRoles, *probe_dcids: str) -> PlaceContext:
    """The place context of a shape bound to `roles`, probed at `probe_dcids`, or at its one
    bound place unexpanded when none are given."""
    return PlaceContext(
        roles=roles, probe_dcids=probe_dcids or None, mode="bound", scope_warning=None
    )


def materialise_alternates(
    shapes: Sequence[ShapeDraft],
    *,
    places: Sequence[PlaceContext],
    elected_bindings: list[SlotBindingDraft],
    elected_has_data: bool,
    graph: Any,
    role_class_walks: RoleClassWalks | None = None,
    **kwargs: Any,
) -> tuple[list[_Candidate], UnsupportedDraft | None]:
    """Elect each alternate under its place context's roles (`elect_alternate`), walked
    through `role_class_walks` when given, and materialise the ones the election keeps, as
    candidate admission does."""
    kept = [
        (alternate, place)
        for shape, place in zip(shapes, places, strict=True)
        if (
            alternate := elect_alternate(
                shape,
                place.roles,
                elected_bindings=elected_bindings,
                elected_has_data=elected_has_data,
                graph=graph,
                role_class_walks=role_class_walks,
            )
        )
        is not None
    ]
    return _materialise_alt_shapes(
        [alternate for alternate, _ in kept],
        places=[place for _, place in kept],
        graph=graph,
        **kwargs,
    )
