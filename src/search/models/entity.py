from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .base import StrictModel
from .grammar import GraphRef
from .identity import IdentityString
from .vocab import Confidence, EntityRoleKey, RoleAxis

# ---------------------------------------------------------------------------
# Entity and EntityRole
# ---------------------------------------------------------------------------


class EntityRoleSubject(StrictModel):
    """The entity is the primary subject of the data."""

    kind: Literal["subject"] = Field(
        default="subject", description="the entity the data is about (default)"
    )


class EntityRoleDirectional(StrictModel):
    """The entity is a directional endpoint (origin or destination) in a flow."""

    kind: Literal["directional"] = Field(
        default="directional", description="directional flow endpoint role discriminator"
    )
    role: GraphRef = Field(
        description=(
            "provenance of how the directional role is sourced "
            "(constraint property for the destination axis, "
            "observation property observationAbout for the origin axis); "
            "consumers read entity.ref for the entity name, not this field"
        )
    )
    axis: RoleAxis = Field(description="origin or destination")


EntityRole = Annotated[
    EntityRoleSubject | EntityRoleDirectional,
    Field(discriminator="kind"),
]


class Entity(StrictModel):
    """A resolved entity the data is about."""

    ref: GraphRef = Field(description="the resolved entity; exists in the graph")
    entity_type: GraphRef | None = Field(
        default=None,
        description="e.g. Country, AdministrativeArea1, Organization",
    )
    role: EntityRole = Field(description="always present; defaults to the subject role")


class ResolvedEntity(StrictModel):
    """One resolved entity on the wire, under `resolved.entities`: the flat
    projection of the engine's own `Entity` -- `ref`/`entity_type` collapse from
    a `GraphRef` to a bare dcid/label pair, and `role` collapses from the
    engine's richer role model to the role-key string it already derives from
    (the identity core's own vocabulary, so this never adds a second mapping)."""

    ref: str = Field(description="the resolved entity's dcid")
    label: str | None = Field(default=None, description="plain-language display name")
    entity_type: str | None = Field(
        default=None,
        description="e.g. Country, AdministrativeArea1, Organization",
    )
    role: EntityRoleKey = Field(
        description="the role key this entity plays; observationAbout, origin, or destination"
    )
    confidence: Confidence | None = Field(
        default=None,
        description=(
            "the engine's confidence in this entity resolution; None when not assessed "
            "-- no entity-resolution confidence signal exists yet, so this ships None "
            "on every entity today"
        ),
    )


_ENTITY_GROUP_LABEL_MAX_LENGTH = 120
"""Code points a group's name may span. Real names ("least developed countries",
"EU institutions") run under 40; the heading and departures echo the label."""


class EntityGroup(StrictModel):
    """A named group whose members fill one entity role, as the request named it
    ("EU member states"). The members themselves are the role's dcids in
    `BaseIdentity.entities`. This records only that they were asked for as one
    group, so the heading can say "the 27 EU member states" after any edit that
    keeps the role's members. A Spec serves its groups on
    `ResolvedSpec.entity_groups`, and a resubmit posts them beside the identity
    (`ResubmitIdentityEntry.entity_groups`)."""

    dcid: IdentityString | None = Field(
        default=None,
        description="the group's own graph node, when the graph holds one; null otherwise",
    )
    label: str = Field(
        min_length=1,
        max_length=_ENTITY_GROUP_LABEL_MAX_LENGTH,
        description="the group's name as the request gave it, plain text",
    )
