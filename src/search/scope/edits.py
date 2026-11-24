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

from code_index.composites import DECLARED_COMPOSITES, provenance_datasets
from code_index.declarations import COMPOSITES
from code_index.measure_basis import edit_basis
from code_index.scope_conventions import BILATERAL_SOURCES, BilateralSource
from code_index.types import CompositeKind
from dfo_platform.roles import role_key_axis
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
    SideBinding,
    SlotBinding,
    YearSpan,
)

BindingKind = Literal["slot", "entities", "shape", "provider", "side"]
IdentityMember = Literal["slots", "entities", "shape", "facets", "requested_window"]

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
    "donor": "side",
    "recipient": "side",
}
"""The binding kind every option of an axis carries."""
PUBLISHED_BINDING: Mapping[ScopeAxis, BindingKind] = {"provider": "slot"}
"""The binding kind an option of an axis may carry beside `AXIS_BINDING`'s on a part that
counts a published total (`tables.AxisTable.published`): a provider option on a published donor
total binds the code of the total that counts its scope (`provider._published_table`)."""


def _with_slot(identity: BaseIdentity, slot: IdentitySlot) -> list[IdentitySlot]:
    """`identity`'s slots with `slot` in place of the slot on its property, or added."""
    kept = [s for s in identity.slots if s.property != slot.property]
    return [*kept, slot]


def _edit_slot(
    identity: BaseIdentity, binding: OptionBinding, siblings: Sequence[OptionBinding]
) -> BaseIdentity:
    """Bind the option's slot to its code or code set, over the option's window when it
    sets one (`SlotBinding.window`)."""
    assert isinstance(binding, SlotBinding)
    bound = (
        IdentityBindingValue(ref=binding.refs[0])
        if binding.state == "value"
        else IdentityBindingSet(refs=list(binding.refs))
    )
    slot = IdentitySlot(property=binding.slot, binding=bound)
    edited = identity.model_copy(update={"slots": _with_slot(identity, slot)})
    return _with_window(edited, binding.window)


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
    """Bind the stacked composite the option names (`ProviderBinding.provenance`) as the
    identity's source, with the option's window when it sets one (`ProviderBinding.window`):
    the bilateral and imputed multilateral spending of the inclusive scope, or the bilateral
    destinations alone of the bilateral scope on a published total that counts core
    contributions (`code_index.scope_conventions.BILATERAL_SOURCES`). Such a composite reads
    a destination total of both sides as the total of the bilateral side
    (`rebound_destinations`). An option naming no composite removes a stacked composite,
    leaving the window as it is: the bilateral scope on a part the CRS serves, and the
    contributions and multilateral scopes, held by the published donor totals that count
    them. Binding and removing are inverses on an identity that binds no other provenance
    and names no destination total of both sides. Raises `ScopeTableError` for an inclusive
    option that names no composite."""
    assert isinstance(binding, ProviderBinding)
    if binding.provenance is None:
        if binding.scope == "inclusive":
            raise ScopeTableError("the inclusive option names no composite source")
        return _with_source(identity, None, kind="stacked")
    edited = _with_source(identity, binding.provenance, kind="stacked")
    return _with_window(rebound_destinations(edited, binding.provenance), binding.window)


def _edit_side(
    identity: BaseIdentity, binding: OptionBinding, siblings: Sequence[OptionBinding]
) -> BaseIdentity:
    """Set the option's role to the parties it names, or remove the role for an option that
    leaves the side open."""
    assert isinstance(binding, SideBinding)
    entities = {key: dcids for key, dcids in identity.entities.items() if key != binding.role}
    if binding.refs:
        entities[binding.role] = list(binding.refs)
    return identity.model_copy(update={"entities": dict(sorted(entities.items()))})


def rebound_destinations(identity: BaseIdentity, provenance: str) -> BaseIdentity:
    """`identity`, whose source is the composite `provenance`, naming the total of the
    bilateral side in place of the total of both sides (`BilateralSource.rebound` for
    `BilateralSource.spanning`) where a component of `provenance` reads the bilateral
    destinations of the dataset the row declares (`ComponentRecipe.destinations`), whose
    population holds no row of the total of both sides. `identity` itself otherwise."""
    composite = DECLARED_COMPOSITES.get(provenance)
    if composite is None:
        return identity
    bound = {
        recipe.dataset for recipe in composite.components if recipe.destinations == "bilateral"
    }
    rebound = {row.spanning: row.rebound for row in BILATERAL_SOURCES if row.dataset in bound}
    entities = dict(identity.entities)
    for key, held in identity.entities.items():
        if role_key_axis(key) == "destination" and rebound.keys() & set(held):
            entities[key] = list(dict.fromkeys(rebound.get(dcid, dcid) for dcid in held))
    if entities == identity.entities:
        return identity
    return identity.model_copy(update={"entities": entities})


def counted_bilateral_source(
    identity: BaseIdentity, *, dataset_of: Callable[[BaseIdentity], str | None]
) -> BilateralSource | None:
    """The bilateral source (`BILATERAL_SOURCES`) whose published total `identity`'s bound
    slots count: the row whose code a slot binds, declared for the dataset `dataset_of`
    elects for `identity`. `dataset_of` runs only where a slot binds a row's code, since
    electing a dataset reads the code index. None where no row is."""
    bound = {
        (slot.property, slot.binding.ref)
        for slot in identity.slots
        if isinstance(slot.binding, IdentityBindingValue)
    }
    sources = [row for row in BILATERAL_SOURCES if (row.code_system, row.code) in bound]
    if not sources:
        return None
    dataset = dataset_of(identity)
    return next((row for row in sources if row.dataset == dataset), None)


def _with_window(identity: BaseIdentity, window: YearSpan | None) -> BaseIdentity:
    """`identity` asking for the years `window` spans, or as it is when `window` is None."""
    if window is None:
        return identity
    first, last = window
    return identity.model_copy(
        update={"requested_window": IdentityWindowRange(start_year=first, end_year=last)}
    )


def _replaceable_pins(provenance: str) -> frozenset[str]:
    """The tables whose pin the source `provenance` replaces: the table every component
    reads, where there is one, and the table a stacked composite's bilateral component
    reads."""
    datasets = provenance_datasets(provenance)
    composite = DECLARED_COMPOSITES.get(provenance)
    pins = set(datasets) if composite is None or len(datasets) == 1 else set()
    if composite is not None and composite.kind == "stacked":
        pins.update(r.dataset for r in composite.components if r.role == "bilateral")
    return frozenset(pins)


def _with_source(
    identity: BaseIdentity, provenance: str | None, *, kind: CompositeKind
) -> BaseIdentity:
    """`identity` binding `provenance` as its source, or, when it is None, without a source
    that binds a declared composite of `kind`. A source binding the one table every component
    of `provenance` reads, or the table a stacked composite's bilateral component reads, gives
    way to it, as DAC1 gives way to the OECD headline that splices DAC1's series
    (`provenance_datasets`) and the CRS to the inclusive composite stacked on it. Raises
    `ScopeTableError` where the identity's source binds anything else, which no option
    replaces."""
    facets = dict(identity.facets or {})
    held = facets.get(_PROVENANCE)
    held_ref = held.ref if isinstance(held, IdentityBindingValue) else None
    held_composite = DECLARED_COMPOSITES.get(held_ref) if held_ref is not None else None
    if held is not None and (held_composite is None or held_composite.kind != kind):
        if provenance is None:
            return identity
        if held_composite is not None or held_ref not in _replaceable_pins(provenance):
            raise ScopeTableError(f"the identity's source binds {held_ref or 'a set'}")
    if provenance is None:
        facets.pop(_PROVENANCE, None)
    else:
        facets[_PROVENANCE] = IdentityBindingValue(ref=provenance)
    return identity.model_copy(update={"facets": facets or None})


EDITS: Mapping[BindingKind, Edit] = {
    "slot": Edit(members=frozenset({"slots", "requested_window"}), apply=_edit_slot),
    "entities": Edit(members=frozenset({"entities"}), apply=_edit_entities),
    "shape": Edit(members=frozenset({"shape", "slots", "facets"}), apply=_edit_shape),
    "provider": Edit(
        members=frozenset({"facets", "entities", "requested_window"}), apply=_edit_provider
    ),
    "side": Edit(members=frozenset({"entities"}), apply=_edit_side),
}
"""The edit of each binding kind."""


def applies_in_pass(binding: OptionBinding) -> bool:
    """Whether a fresh search applies an option binding `binding` to the identity it
    resolved: it binds a declared composite as the identity's source, the OECD headline
    basis (`ShapeBinding.provenance`) or a provider scope's composite, or it sets a provider
    scope on the identity's source, as the bilateral scope on the CRS removes the inclusive
    composite. Any other option reads another identity, which a switch or resubmit serves."""
    return isinstance(binding, ProviderBinding) or (
        isinstance(binding, ShapeBinding) and binding.provenance is not None
    )


def apply_binding(
    identity: BaseIdentity, binding: OptionBinding, siblings: Sequence[OptionBinding]
) -> BaseIdentity:
    """`identity` edited onto `binding`, among its axis's `siblings`. Raises
    `ScopeTableError` for a binding kind with no edit."""
    edit = EDITS.get(binding.kind)
    if edit is None:
        raise ScopeTableError(f"no identity edit is declared for a {binding.kind!r} binding")
    return edit.apply(identity, binding, siblings)
