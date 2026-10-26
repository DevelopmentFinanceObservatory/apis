"""Sets an identity document's measure basis from the declared basis measures.

Search's measure-basis option (`search.scope.edits`) and the OECD headline composite's
components (`code_index.composites`) both edit an identity's measure here, so a basis binds the
same shape and slots wherever it is chosen.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from code_index.scope_conventions import BASIS_MEASURES, ODA_FLOW, ODA_FLOW_SYSTEM, BasisMeasure

_NETTING_BASIS = "nettingBasis"
_FLOW_TYPE_PROPERTIES = frozenset(row.flow_type[0] for row in BASIS_MEASURES)
_BASIS_POPULATIONS = frozenset(row.population_type for row in BASIS_MEASURES)
_DESCRIBED = "statvar_schema"

type IdentitySlotDocument = Mapping[str, object]


def declared_basis(measured_property: str, netting_basis: str | None) -> BasisMeasure | None:
    """The declared basis (`scope_conventions.BASIS_MEASURES`) whose measure and netting basis
    are `measured_property` and `netting_basis`, or `None`."""
    return next(
        (
            row
            for row in BASIS_MEASURES
            if (row.measured_property, row.netting_basis) == (measured_property, netting_basis)
        ),
        None,
    )


def edit_basis(
    identity: Mapping[str, object], *, measured_property: str, netting_basis: str | None
) -> dict[str, object]:
    """`identity`, a described identity document, measured as `measured_property` with its
    netting-basis slot bound to `netting_basis`, or removed when it is `None`.

    On an identity whose population a declared basis measures, the basis the two name
    (`declared_basis`) also sets the population, the flow-type slot that says a flow is
    disbursed or committed, and the CRS types of flow an ODA flow binding reads as
    (`BasisMeasure.oda_flows`). The result is uncanonicalised: an edited slot moves to the end.
    Raises `ValueError` for a pinned shape, which has no measure to set."""
    shape = cast("Mapping[str, object]", identity["shape"])
    if shape.get("kind") != _DESCRIBED:
        raise ValueError("a pinned shape has no measure to set")
    basis = declared_basis(measured_property, netting_basis)
    update: dict[str, object] = {"measured_property": measured_property}
    if basis is not None and shape.get("population_type") in _BASIS_POPULATIONS:
        update["population_type"] = basis.population_type
    else:
        basis = None
    slots = list(cast("Sequence[IdentitySlotDocument]", identity.get("slots", ())))
    if netting_basis is None:
        slots = [slot for slot in slots if slot["property"] != _NETTING_BASIS]
    else:
        slots = with_slot(
            slots,
            {"property": _NETTING_BASIS, "binding": {"state": "value", "ref": netting_basis}},
        )
    if basis is not None:
        slots = _basis_slots(slots, basis)
    return {**identity, "shape": {**shape, **update}, "slots": slots}


def with_slot(
    slots: Sequence[IdentitySlotDocument], slot: IdentitySlotDocument
) -> list[IdentitySlotDocument]:
    """`slots` with `slot` in place of the slot on its property, or added."""
    kept = [s for s in slots if s["property"] != slot["property"]]
    return [*kept, slot]


def _basis_slots(
    slots: Sequence[IdentitySlotDocument], basis: BasisMeasure
) -> list[IdentitySlotDocument]:
    """`slots` holding `basis`'s flow type in place of any other the slots bind, and its ODA
    flows in place of an ODA flow binding: CRS flow 10, alone or beside the other flows a
    basis reads it as."""
    flow_type, flow_value = basis.flow_type
    oda_readings = {frozenset(row.oda_flows) for row in BASIS_MEASURES}
    edited: list[IdentitySlotDocument] = []
    for slot in slots:
        if slot["property"] in _FLOW_TYPE_PROPERTIES:
            slot = {"property": flow_type, "binding": {"state": "value", "ref": flow_value}}
        elif slot["property"] == ODA_FLOW_SYSTEM and bound_refs(slot) in oda_readings:
            slot = {
                "property": slot["property"],
                "binding": (
                    {"state": "value", "ref": ODA_FLOW}
                    if basis.oda_flows == (ODA_FLOW,)
                    else {"state": "set", "refs": list(basis.oda_flows)}
                ),
            }
        edited.append(slot)
    return edited


def bound_refs(slot: IdentitySlotDocument) -> frozenset[str]:
    """The codes `slot` binds as a value or a set, empty for an open or exact slot."""
    binding = cast("Mapping[str, object]", slot["binding"])
    if binding.get("state") == "value":
        return frozenset((cast(str, binding["ref"]),))
    if binding.get("state") == "set":
        return frozenset(cast("Sequence[str]", binding["refs"]))
    return frozenset()
