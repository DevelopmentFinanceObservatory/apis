"""Build verified `(base_spec_id, BaseIdentity)` fixtures through the production identity hash."""

from __future__ import annotations

from collections.abc import Mapping

from search.engine.shape import identity_shape_for
from search.engine.spec_id import build_spec_identity, entity_role_key
from search.models import (
    BaseIdentity,
    BindingValue,
    Entity,
    EntityRoleDirectional,
    EntityRoleSubject,
    GraphRef,
    IdentityFacetBinding,
    RequestedWindow,
    Slot,
    SlotKey,
    SlotValue,
)

# `entity_role_key` derives the recipient role; this GraphRef is provenance-only.
_RECIPIENT_ROLE_DCID = "DevelopmentFinanceRecipient"
_DONOR_ROLE_DCID = "observationAbout"

# A fixture-only shape for tests that need a stable described-arm identity.
DECLARING_SHAPE: dict = identity_shape_for(
    pop_type_dcid="CreditorDebtorFlow",
    meas_prop_dcid="flowAmount",
    stat_type_dcid="measuredValue",
    meas_qual_dcid=None,
    meas_denom_dcid=None,
)

# Real GDP shape for standard-promotion fixtures; resubmit re-derives it from graph arcs.
# A placeholder shape fails the structural match.
GDP_SHAPE: dict = {
    "kind": "statvar_schema",
    "population_type": "EconomicActivity",
    "measured_property": "amount",
    "stat_type": "measuredValue",
    "measurement_qualifier": "Nominal",
}

# Real `Count_Person` shape for the same structural resubmit check.
PERSON_COUNT_SHAPE: dict = {
    "kind": "statvar_schema",
    "population_type": "Person",
    "measured_property": "count",
    "stat_type": "measuredValue",
}


def shape_for(name: str) -> dict:
    """Build an identity shape whose `population_type` distinguishes fixtures by name."""
    return {"kind": "statvar_schema", "population_type": name}


def make_base(
    shape: Mapping[str, object],
    slots: list[Slot] | None = None,
    *,
    entities: list[Entity] | None = None,
    requested_window: RequestedWindow | None = None,
    explicit_facets: Mapping[str, IdentityFacetBinding] | None = None,
) -> tuple[str, BaseIdentity]:
    """Build a resubmit identity pair through the production hash.

    `shape` is the wire-form identity; slots, entities, window, and facets feed the hash.
    `resolved_stat_var_dcids` is graph state and belongs on the payload separately.
    """
    slots = slots or []
    return build_spec_identity(
        shape,
        slots,
        entities=entities,
        requested_window=requested_window,
        explicit_facets=explicit_facets,
        open_roles=(),
    )


def make_identity(
    shape: Mapping[str, object],
    slots: list[Slot] | None = None,
    *,
    entities: list[Entity] | None = None,
    requested_window: RequestedWindow | None = None,
    explicit_facets: Mapping[str, IdentityFacetBinding] | None = None,
) -> BaseIdentity:
    """Build a posted ResubmitIdentityPayload identity with the production projection."""
    return make_base(
        shape,
        slots,
        entities=entities,
        requested_window=requested_window,
        explicit_facets=explicit_facets,
    )[1]


def subject_entity(dcid: str, label: str | None = None) -> Entity:
    """An observationAbout-role Entity, for a standard shape's base_identity.entities."""
    return Entity(ref=GraphRef(ref=dcid, label=label or dcid), role=EntityRoleSubject())


def destination_entity(dcid: str, label: str | None = None) -> Entity:
    """A destination-axis (recipient)-role Entity, for a dev-finance base_identity.entities."""
    return Entity(
        ref=GraphRef(ref=dcid, label=label or dcid),
        role=EntityRoleDirectional(
            role=GraphRef(ref=_RECIPIENT_ROLE_DCID, label="recipient"), axis="destination"
        ),
    )


def base_from_spec(spec) -> tuple[str, BaseIdentity]:
    """Return `(spec.spec_id, spec.definition.identity)` for resubmission.

    Standard-shape callers set `resolved_stat_var_dcids` from `spec.stat_vars` separately.
    """
    return spec.spec_id, spec.definition.identity


def where_slots_for(entities: list[Entity] | None) -> list[Slot]:
    """Build legacy `where` slots for wire-adapter fixtures.

    Resubmit payloads carry entities inside `BaseIdentity`.
    """
    slots: list[Slot] = []
    for entity in entities or []:
        role = entity_role_key(entity)
        if role == "destination":
            key = SlotKey(
                axis="where",
                property=GraphRef(ref=_RECIPIENT_ROLE_DCID, label="recipient"),
                label="recipient",
            )
        elif role == "observationAbout":
            key = SlotKey(axis="where", property=None, label="place")
        else:
            continue
        slots.append(
            Slot(
                key=key, binding=BindingValue(value=SlotValue(ref=entity.ref, value_kind="entity"))
            )
        )
    return slots


def origin_entity(dcid: str, label: str | None = None) -> Entity:
    """An origin-axis (provider/donor)-role Entity, for a dev-finance base_identity.entities."""
    return Entity(
        ref=GraphRef(ref=dcid, label=label or dcid),
        role=EntityRoleDirectional(
            role=GraphRef(ref=_DONOR_ROLE_DCID, label="donor"), axis="origin"
        ),
    )
