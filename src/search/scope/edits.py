"""The identity edit each scope option makes.

`AXIS_BINDING` names the binding kind an axis's options carry, and `EDITS` the edit of each
binding kind: which identity members it touches and how. An edit takes the identity, the
option's binding and every binding of the option's axis, so an option replaces whatever a
sibling option put there and the edit needs no record of the option it replaces. An edit
returns the identity uncanonicalised; `compile` mints the result, which sorts, de-duplicates
and collapses singleton sets.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from code_index.scope_conventions import BASIS_MEASURES, ODA_FLOW, ODA_FLOW_SYSTEM, BasisMeasure
from search.models import (
    BaseIdentity,
    EntitiesBinding,
    IdentityBindingSet,
    IdentityBindingValue,
    IdentityShapeSchema,
    IdentitySlot,
    OptionBinding,
    ScopeAxis,
    ShapeBinding,
    SlotBinding,
)

BindingKind = Literal["slot", "entities", "shape", "provider"]
IdentityMember = Literal["slots", "entities", "shape", "facets"]

_NETTING_BASIS = "nettingBasis"
_FLOW_TYPE_PROPERTIES = frozenset(row.flow_type[0] for row in BASIS_MEASURES)
_BASIS_POPULATIONS = frozenset(row.population_type for row in BASIS_MEASURES)


class ScopeTableError(ValueError):
    """An axis table or option that no edit can apply: a programming error in a rule."""


@dataclass(frozen=True)
class Edit:
    """The identity members an edit touches and the edit itself."""

    members: frozenset[IdentityMember]
    apply: Callable[[BaseIdentity, OptionBinding, Sequence[OptionBinding]], BaseIdentity]


AXIS_BINDING: Mapping[ScopeAxis, BindingKind] = {
    "sector": "slot",
    "region": "entities",
    "measure_basis": "shape",
    "provider": "provider",
}
"""The binding kind every option of an axis carries."""


def _with_slot(identity: BaseIdentity, slot: IdentitySlot) -> list[IdentitySlot]:
    """`identity`'s slots with `slot` in place of the slot on its property, or added."""
    kept = [s for s in identity.slots if s.property != slot.property]
    return [*kept, slot]


def _edit_slot(
    identity: BaseIdentity, binding: OptionBinding, siblings: Sequence[OptionBinding]
) -> BaseIdentity:
    """Bind the option's slot to its code or code set."""
    assert isinstance(binding, SlotBinding)
    bound = (
        IdentityBindingValue(ref=binding.refs[0])
        if binding.state == "value"
        else IdentityBindingSet(refs=list(binding.refs))
    )
    slot = IdentitySlot(property=binding.slot, binding=bound)
    return identity.model_copy(update={"slots": _with_slot(identity, slot)})


def _edit_entities(
    identity: BaseIdentity, binding: OptionBinding, siblings: Sequence[OptionBinding]
) -> BaseIdentity:
    """Remove from the option's role every entity a sibling option adds, then add the
    option's own. Raises `ScopeTableError` when the role would hold none."""
    assert isinstance(binding, EntitiesBinding)
    sibling_adds = {
        dcid
        for sibling in siblings
        if isinstance(sibling, EntitiesBinding) and sibling.role == binding.role
        for dcid in sibling.add
    }
    held = [dcid for dcid in identity.entities.get(binding.role, []) if dcid not in sibling_adds]
    entities = [*held, *(dcid for dcid in binding.add if dcid not in held)]
    if not entities:
        raise ScopeTableError(f"the option would leave {binding.role!r} with no entity")
    return identity.model_copy(update={"entities": {**identity.entities, binding.role: entities}})


def _edit_shape(
    identity: BaseIdentity, binding: OptionBinding, siblings: Sequence[OptionBinding]
) -> BaseIdentity:
    """Set the measured property, and bind the netting basis, or drop it for a measure
    that carries none. Raises `ScopeTableError` for a pinned shape.

    On an identity whose population a declared basis measures
    (`scope_conventions.BASIS_MEASURES`), the basis the binding names also sets the
    population, the flow-type slot that says a flow is disbursed or committed, and the CRS
    types of flow an ODA flow binding reads as (`BasisMeasure.oda_flows`)."""
    assert isinstance(binding, ShapeBinding)
    if not isinstance(identity.shape, IdentityShapeSchema):
        raise ScopeTableError("a pinned shape has no measure to set")
    basis = _declared_basis(binding)
    update: dict[str, str] = {"measured_property": binding.measured_property}
    if basis is not None and identity.shape.population_type in _BASIS_POPULATIONS:
        update["population_type"] = basis.population_type
    else:
        basis = None
    shape = identity.shape.model_copy(update=update)
    if binding.netting_basis is None:
        slots = [s for s in identity.slots if s.property != _NETTING_BASIS]
    else:
        netting = IdentitySlot(
            property=_NETTING_BASIS, binding=IdentityBindingValue(ref=binding.netting_basis)
        )
        slots = _with_slot(identity, netting)
    if basis is not None:
        slots = _basis_slots(slots, basis)
    return identity.model_copy(update={"shape": shape, "slots": slots})


def _declared_basis(binding: ShapeBinding) -> BasisMeasure | None:
    """The declared basis whose measure and netting basis `binding` binds."""
    return next(
        (
            row
            for row in BASIS_MEASURES
            if (row.measured_property, row.netting_basis)
            == (binding.measured_property, binding.netting_basis)
        ),
        None,
    )


def _basis_slots(slots: Sequence[IdentitySlot], basis: BasisMeasure) -> list[IdentitySlot]:
    """`slots` holding `basis`'s flow type in place of any other the slots bind, and its ODA
    flows in place of an ODA flow binding: CRS flow 10, alone or beside the other flows a
    basis reads it as."""
    flow_type, flow_value = basis.flow_type
    oda_readings = {frozenset(row.oda_flows) for row in BASIS_MEASURES}
    edited: list[IdentitySlot] = []
    for slot in slots:
        if slot.property in _FLOW_TYPE_PROPERTIES:
            slot = IdentitySlot(property=flow_type, binding=IdentityBindingValue(ref=flow_value))
        elif slot.property == ODA_FLOW_SYSTEM and _bound_refs(slot) in oda_readings:
            slot = IdentitySlot(
                property=slot.property,
                binding=(
                    IdentityBindingValue(ref=ODA_FLOW)
                    if basis.oda_flows == (ODA_FLOW,)
                    else IdentityBindingSet(refs=list(basis.oda_flows))
                ),
            )
        edited.append(slot)
    return edited


def _bound_refs(slot: IdentitySlot) -> frozenset[str]:
    """The codes `slot` binds, empty for an open slot."""
    if isinstance(slot.binding, IdentityBindingValue):
        return frozenset((slot.binding.ref,))
    if isinstance(slot.binding, IdentityBindingSet):
        return frozenset(slot.binding.refs)
    return frozenset()


EDITS: Mapping[BindingKind, Edit] = {
    "slot": Edit(members=frozenset({"slots"}), apply=_edit_slot),
    "entities": Edit(members=frozenset({"entities"}), apply=_edit_entities),
    "shape": Edit(members=frozenset({"shape", "slots"}), apply=_edit_shape),
}
"""The edit of each binding kind. A provider binding has none until the composite of
bilateral and imputed multilateral spending is declared as a provenance."""


def apply_binding(
    identity: BaseIdentity, binding: OptionBinding, siblings: Sequence[OptionBinding]
) -> BaseIdentity:
    """`identity` edited onto `binding`, among its axis's `siblings`. Raises
    `ScopeTableError` for a binding kind with no edit."""
    edit = EDITS.get(binding.kind)
    if edit is None:
        raise ScopeTableError(f"no identity edit is declared for a {binding.kind!r} binding")
    return edit.apply(identity, binding, siblings)
