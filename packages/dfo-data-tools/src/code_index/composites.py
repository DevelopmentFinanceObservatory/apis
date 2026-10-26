"""Derives a composite identity's component identities.

An identity is a composite when its `facets.provenance` binds a declared composite provenance
(`code_index.declarations.COMPOSITES`) as a value. No reader takes such an identity's
provenance pin as a dataset: `component_identities` is the one derivation of what each
component reads, and Search, its resubmit and the Datasets mint all call it before any read.
`code_index.members.pinned_refs` refuses a composite ref (`CompositeProvenanceRead`), so a
reader that skips the derivation fails rather than reading the node nobody publishes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from code_index.declarations import COMPOSITES
from code_index.measure_basis import edit_basis, with_slot
from code_index.types import (
    ComponentRecipe,
    ComponentRole,
    CompositeDeclaration,
    StatVarRow,
    YearPeriod,
)

if TYPE_CHECKING:
    from code_index.query import CodeIndex

_PROVENANCE = "provenance"
_PINNED_SHAPE = "stat_var"

DECLARED_COMPOSITES: Mapping[str, CompositeDeclaration] = {
    composite.provenance: composite for composite in COMPOSITES
}
"""Every declared composite by its provenance."""


class CompositeNotServable(ValueError):
    """A composite identity whose components cannot be derived: it binds a slot a component's
    rows do not declare, pins a StatVar, names a composite in a provenance set, or names one
    the index serves no component table of. `ref` is the composite's provenance and `pointer`
    the identity member at fault, relative to the identity."""

    def __init__(self, message: str, *, ref: str, pointer: str) -> None:
        self.ref = ref
        self.pointer = pointer
        super().__init__(message)


class CompositeProvenanceRead(ValueError):
    """A provenance pin read as a dataset when it binds a composite. The caller reads the
    composite's components (`component_identities`) instead."""


@dataclass(frozen=True)
class ComponentIdentity:
    """One component of a composite identity: the role it plays, the recipe it was derived
    by, and its own identity document, which reads the component's table alone."""

    role: ComponentRole
    recipe: ComponentRecipe
    identity: Mapping[str, object]


def composite_ref(identity: Mapping[str, object]) -> str | None:
    """The declared composite `identity`'s provenance facet binds, or `None` when it binds
    none.

    Raises `CompositeNotServable` for a composite named in a provenance set, which would read
    as an election among the set's tables."""
    facets = cast("Mapping[str, Mapping[str, object]]", identity.get("facets") or {})
    binding = facets.get(_PROVENANCE, {})
    ref = binding.get("ref")
    if isinstance(ref, str):
        return ref if ref in DECLARED_COMPOSITES else None
    refs = cast("Sequence[str]", binding.get("refs", ()))
    named = [ref for ref in refs if ref in DECLARED_COMPOSITES]
    if named:
        raise CompositeNotServable(
            f"provenance set {list(refs)!r} names the composite "
            f"{named[0]!r}, which is bound as a value alone",
            ref=named[0],
            pointer=f"/facets/{_PROVENANCE}",
        )
    return None


def refuse_composite_refs(refs: Sequence[str]) -> None:
    """Raise `CompositeProvenanceRead` when `refs`, a provenance pin, names a declared
    composite."""
    named = [ref for ref in refs if ref in DECLARED_COMPOSITES]
    if named:
        raise CompositeProvenanceRead(
            f"provenance {named[0]!r} is a composite; read its components "
            "(code_index.composites.component_identities) instead of the pin"
        )


def component_identities(
    index: CodeIndex, identity: Mapping[str, object]
) -> tuple[ComponentIdentity, ...] | None:
    """The component identities of a composite `identity`, in declaration order, or `None`
    for an identity that binds no composite.

    Each component's identity is `identity` with the composite's provenance binding replaced
    by the recipe's own (`ComponentRecipe.provenance`), its measure edited as the recipe says,
    the recipe's slots bound and its undeclared slots removed, and its window clipped to the
    recipe's periods. A spliced component whose periods the window does not reach is left out.
    Slots stay sorted by property.

    Raises `CompositeNotServable` when `identity` pins a StatVar, binds a slot a component
    drops (`ComponentRecipe.drop_unbound`), names a composite in a provenance set, names one
    whose component tables `index` does not serve (`CodeIndex.composite`), or reaches none of
    its components' periods."""
    ref = composite_ref(identity)
    if ref is None:
        return None
    declaration = index.composite(ref)
    if declaration is None:
        raise CompositeNotServable(
            f"the code index serves no component table of composite {ref!r}",
            ref=ref,
            pointer=f"/facets/{_PROVENANCE}",
        )
    shape = cast("Mapping[str, object]", identity.get("shape", {}))
    if shape.get("kind") == _PINNED_SHAPE:
        raise CompositeNotServable(
            f"composite {ref!r} composes described identities; a pinned StatVar has no components",
            ref=ref,
            pointer="/shape",
        )
    window = cast("Mapping[str, object]", identity.get("requested_window", {"kind": "all"}))
    components: list[ComponentIdentity] = []
    for recipe in declaration.components:
        clipped = _clipped_window(window, recipe.periods)
        if clipped is None:
            continue
        derived = _derived_identity(identity, recipe, ref=ref)
        components.append(
            ComponentIdentity(
                role=recipe.role,
                recipe=recipe,
                identity={**derived, "requested_window": clipped},
            )
        )
    if not components:
        raise CompositeNotServable(
            f"the requested window reaches none of composite {ref!r}'s component periods",
            ref=ref,
            pointer="/requested_window",
        )
    return tuple(components)


def component_of(composite: CompositeDeclaration, row: StatVarRow) -> ComponentRole | None:
    """The role of `composite`'s component that serves `row`: the first whose table publishes
    it (`StatVarRow.provenances`) and whose recipe sets no measure or the measure `row` has.
    `None` when no component serves it."""
    for recipe in composite.components:
        measure = recipe.basis[0] if recipe.basis is not None else recipe.measured_property
        if recipe.dataset in row.provenances and measure in (None, row.measured_property):
            return recipe.role
    return None


def without_composite(identity: Mapping[str, object]) -> Mapping[str, object]:
    """`identity` with a composite provenance binding removed: its own shape, slots, entities
    and window, read as an identity that pins no provenance. `identity` itself when it binds
    no composite."""
    if composite_ref(identity) is None:
        return identity
    facets = {
        axis: binding
        for axis, binding in cast("Mapping[str, object]", identity["facets"]).items()
        if axis != _PROVENANCE
    }
    stripped = {key: value for key, value in identity.items() if key != "facets"}
    return {**stripped, "facets": facets} if facets else stripped


def _derived_identity(
    identity: Mapping[str, object], recipe: ComponentRecipe, *, ref: str
) -> dict[str, object]:
    """`identity` derived by `recipe`, its window unchanged."""
    derived = dict(without_composite(identity))
    if recipe.provenance is not None:
        facets = dict(cast("Mapping[str, object]", derived.get("facets", {})))
        facets[_PROVENANCE] = {"state": "value", "ref": recipe.provenance}
        derived["facets"] = facets
    if recipe.basis is not None:
        measured_property, netting_basis = recipe.basis
        derived = edit_basis(
            derived, measured_property=measured_property, netting_basis=netting_basis
        )
    elif recipe.measured_property is not None:
        shape = cast("Mapping[str, object]", derived["shape"])
        derived["shape"] = {**shape, "measured_property": recipe.measured_property}
    slots = list(cast("Sequence[Mapping[str, object]]", derived.get("slots", ())))
    for position, slot in enumerate(slots):
        binding = cast("Mapping[str, object]", slot["binding"])
        if slot["property"] in recipe.drop_unbound and binding.get("state") != "unbound":
            raise CompositeNotServable(
                f"composite {ref!r} serves no {slot['property']!r} binding: its "
                f"{recipe.role} component's rows do not declare it",
                ref=ref,
                pointer=f"/slots/{position}",
            )
    slots = [slot for slot in slots if slot["property"] not in recipe.drop_unbound]
    for prop, code in sorted(recipe.bind.items()):
        slots = with_slot(slots, {"property": prop, "binding": {"state": "value", "ref": code}})
    derived["slots"] = sorted(slots, key=lambda slot: cast(str, slot["property"]))
    return derived


def _clipped_window(
    window: Mapping[str, object], periods: YearPeriod | None
) -> Mapping[str, object] | None:
    """`window` clipped to `periods`, or `None` when the two share no year. A `latest`
    window passes unclipped, as does any window when `periods` is `None`."""
    if periods is None or window.get("kind") == "latest":
        return window
    start = cast("int | None", window.get("start_year")) if window.get("kind") == "range" else None
    end = cast("int | None", window.get("end_year")) if window.get("kind") == "range" else None
    first = _later(start, periods.first)
    last = _earlier(end, periods.last)
    if first is not None and last is not None and first > last:
        return None
    if first is None and last is None:
        return {"kind": "all"}
    clipped: dict[str, object] = {"kind": "range"}
    if first is not None:
        clipped["start_year"] = first
    if last is not None:
        clipped["end_year"] = last
    return clipped


def _later(a: int | None, b: int | None) -> int | None:
    return b if a is None else a if b is None else max(a, b)


def _earlier(a: int | None, b: int | None) -> int | None:
    return b if a is None else a if b is None else min(a, b)
