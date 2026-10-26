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

from code_index.composites import DECLARED_COMPOSITES
from code_index.declarations import COMPOSITES
from code_index.measure_basis import edit_basis
from code_index.types import CompositeKind
from search.models import (
    BaseIdentity,
    EntitiesBinding,
    IdentityBindingSet,
    IdentityBindingValue,
    IdentityShapeSchema,
    IdentitySlot,
    IdentityWindowRange,
    OptionBinding,
    ProviderBinding,
    ScopeAxis,
    ShapeBinding,
    SlotBinding,
)

BindingKind = Literal["slot", "entities", "shape", "provider"]
IdentityMember = Literal["slots", "entities", "shape", "facets", "requested_window"]

INCLUSIVE_PROVENANCE: str = next(c.provenance for c in COMPOSITES if c.kind == "stacked")
"""The composite of a donor's bilateral flows and its imputed multilateral share, which the
inclusive provider scope binds as the identity's source."""
HEADLINE_PROVENANCE: str = next(c.provenance for c in COMPOSITES if c.kind == "spliced")
"""The OECD headline composite, net flows to 2017 and grant equivalents from 2018, which the
`oecd_headline` measure basis binds as the identity's source."""

_PROVENANCE = "provenance"


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
    that carries none (`code_index.measure_basis.edit_basis`). Bind the option's composite
    provenance as the identity's source (`ShapeBinding.provenance`, the OECD headline), or
    remove a spliced composite another basis option bound. Raises `ScopeTableError` for a
    pinned shape."""
    assert isinstance(binding, ShapeBinding)
    if not isinstance(identity.shape, IdentityShapeSchema):
        raise ScopeTableError("a pinned shape has no measure to set")
    edited = edit_basis(
        identity.model_dump(mode="json"),
        measured_property=binding.measured_property,
        netting_basis=binding.netting_basis,
    )
    return _with_source(BaseIdentity.model_validate(edited), binding.provenance, kind="spliced")


def _edit_provider(
    identity: BaseIdentity, binding: OptionBinding, siblings: Sequence[OptionBinding]
) -> BaseIdentity:
    """Bind the stacked composite of bilateral and imputed multilateral spending as the
    identity's source for the inclusive scope, with the option's window when it sets one
    (`ProviderBinding.window`), or remove that composite for the bilateral scope, leaving the
    window as it is. The two are inverses on an identity that binds no other provenance."""
    assert isinstance(binding, ProviderBinding)
    if binding.scope == "bilateral":
        return _with_source(identity, None, kind="stacked")
    edited = _with_source(identity, INCLUSIVE_PROVENANCE, kind="stacked")
    if binding.window is None:
        return edited
    first, last = binding.window
    return edited.model_copy(
        update={"requested_window": IdentityWindowRange(start_year=first, end_year=last)}
    )


def _with_source(
    identity: BaseIdentity, provenance: str | None, *, kind: CompositeKind
) -> BaseIdentity:
    """`identity` binding `provenance` as its source, or, when it is None, without a source
    that binds a declared composite of `kind`. Raises `ScopeTableError` where the identity's
    source binds anything else, which no option replaces."""
    facets = dict(identity.facets or {})
    held = facets.get(_PROVENANCE)
    held_ref = held.ref if isinstance(held, IdentityBindingValue) else None
    held_composite = DECLARED_COMPOSITES.get(held_ref) if held_ref is not None else None
    if held is not None and (held_composite is None or held_composite.kind != kind):
        if provenance is None:
            return identity
        raise ScopeTableError(f"the identity's source binds {held_ref or 'a set'}")
    if provenance is None:
        facets.pop(_PROVENANCE, None)
    else:
        facets[_PROVENANCE] = IdentityBindingValue(ref=provenance)
    return identity.model_copy(update={"facets": facets or None})


EDITS: Mapping[BindingKind, Edit] = {
    "slot": Edit(members=frozenset({"slots"}), apply=_edit_slot),
    "entities": Edit(members=frozenset({"entities"}), apply=_edit_entities),
    "shape": Edit(members=frozenset({"shape", "slots", "facets"}), apply=_edit_shape),
    "provider": Edit(members=frozenset({"facets", "requested_window"}), apply=_edit_provider),
}
"""The edit of each binding kind."""


def apply_binding(
    identity: BaseIdentity, binding: OptionBinding, siblings: Sequence[OptionBinding]
) -> BaseIdentity:
    """`identity` edited onto `binding`, among its axis's `siblings`. Raises
    `ScopeTableError` for a binding kind with no edit."""
    edit = EDITS.get(binding.kind)
    if edit is None:
        raise ScopeTableError(f"no identity edit is declared for a {binding.kind!r} binding")
    return edit.apply(identity, binding, siblings)
