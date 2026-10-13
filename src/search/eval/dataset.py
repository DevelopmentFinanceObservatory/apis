"""Build eval requests and Langfuse item ids from cases.

`resubmit_identity` builds a dataset_resubmit case's identity, which the runner posts.
`item_id_for_case` names a case's Langfuse dataset item.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

from search.engine.spec_id import build_spec_identity
from search.eval.cases import Case, Resubmit
from search.models import (
    BaseIdentity,
    BindingSet,
    BindingUnbound,
    BindingValue,
    Entity,
    EntityRoleDirectional,
    EntityRoleSubject,
    GraphRef,
    IdentityBindingSet,
    IdentityBindingValue,
    IdentityFacetBinding,
    Slot,
    SlotKey,
    SlotValue,
    TimeWindow,
)


def _requested_window(slots: Iterable[Slot]) -> TimeWindow | None:
    """Return the window a bound ``when`` slot carries, or None when it names none.

    Identity hashing excludes the ``when`` axis; the window rides on ``requested_window``.
    """
    for slot in slots:
        binding = slot.binding
        if (
            slot.key.axis == "when"
            and isinstance(binding, BindingValue)
            and binding.value.value_kind == "time_window"
        ):
            return binding.value.time_window
    return None


def resubmit_identity(resubmit: Resubmit) -> BaseIdentity:
    """Build a dataset_resubmit case's identity.

    Subject entities key under `observationAbout`. A case naming origins keys its
    entities under `destination` and its origins under `origin`. A bound ``when`` slot's
    window becomes the identity's ``requested_window``. A case opens no role. Raises ``ValueError`` when a
    what or how slot carries a literal or time-window value no identity holds.
    """
    slots, explicit_facets = _legacy_slots_with_provenance(resubmit.slots, resubmit.facets)
    entities: list[Entity] = []
    if not resubmit.origin_dcids:
        entities.extend(
            Entity(ref=GraphRef(ref=dcid, label=dcid), role=EntityRoleSubject())
            for dcid in resubmit.entity_dcids or []
        )
    else:
        entities.extend(
            Entity(ref=GraphRef(ref=dcid, label=dcid), role=_directional(axis))
            for axis, dcids in (
                ("destination", resubmit.entity_dcids or []),
                ("origin", resubmit.origin_dcids),
            )
            for dcid in dcids
        )
    try:
        _spec_id, identity = build_spec_identity(
            resubmit.shape.model_dump(mode="json", exclude_unset=True),
            slots,
            entities=entities,
            open_roles=(),
            requested_window=_requested_window(resubmit.slots),
            explicit_facets=explicit_facets,
        )
    except ValueError as exc:
        if "must never reach identity" not in str(exc):
            raise
        raise ValueError(f"dataset_resubmit needs an identity definition: {exc}") from exc
    return identity


def _directional(axis: Literal["destination", "origin"]) -> EntityRoleDirectional:
    """A directional role named by its own axis, as legacy cases carry no role names."""
    return EntityRoleDirectional(role=GraphRef(ref=axis, label=axis), axis=axis)


def _legacy_slots_with_provenance(
    slots: list[Slot],
    facets: Mapping[str, IdentityFacetBinding] | None,
) -> tuple[list[Slot], dict[str, IdentityFacetBinding] | None]:
    """Move legacy ``facets.provenance`` onto the canonical source slot.

    ``build_spec_identity`` hashes provenance only from a source-axis wire Slot. A legacy
    case can carry both forms, so this applies the same single-channel rule before minting.
    A bound source slot beside a provenance facet is a contradiction and raises
    ``ValueError``. An unbound source slot takes the provenance, and a record with no source
    slot gets one appended. ``measurementMethod`` and ``unit`` stay explicit facets.
    """
    provenance = (facets or {}).get("provenance")
    explicit_facets = {
        axis: binding for axis, binding in (facets or {}).items() if axis != "provenance"
    }
    if provenance is None:
        return slots, explicit_facets or None

    source_slots = [slot for slot in slots if slot.key.axis == "source"]
    if any(not isinstance(slot.binding, BindingUnbound) for slot in source_slots):
        raise ValueError("a source-axis slot and facets.provenance cannot both be bound")

    if isinstance(provenance, IdentityBindingValue):
        values = [
            SlotValue(ref=GraphRef(ref=provenance.ref, label=provenance.ref), value_kind="source")
        ]
    elif isinstance(provenance, IdentityBindingSet):
        values = [
            SlotValue(ref=GraphRef(ref=dcid, label=dcid), value_kind="source")
            for dcid in provenance.refs
        ]
    else:  # IdentityFacetBinding excludes unbound and request-form provenance.
        raise ValueError(f"invalid provenance facet binding: {type(provenance)!r}")

    binding = BindingValue(value=values[0]) if len(values) == 1 else BindingSet(values=values)
    if source_slots:
        return (
            [
                slot.model_copy(update={"binding": binding}) if slot.key.axis == "source" else slot
                for slot in slots
            ],
            explicit_facets or None,
        )
    return (
        [
            *slots,
            Slot(
                key=SlotKey(axis="source", property=None, label="data source"),
                binding=binding,
            ),
        ],
        explicit_facets or None,
    )


def item_id_for_case(case: Case, dataset_name: str) -> str:
    """Return a case's item id, scoped by dataset name because Langfuse ids are global."""
    return f"{dataset_name}:search-case-{case.id}"
