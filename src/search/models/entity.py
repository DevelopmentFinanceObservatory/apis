from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .base import StrictModel
from .grammar import GraphRef
from .identity import IdentityString
from .vocab import Confidence, EntityRoleKey, PartyKind, RoleAxis

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


def _is_none(value: object) -> bool:
    return value is None


_ENTITY_GROUP_LABEL_MAX_LENGTH = 120
"""Code points a group's name may span. Real names ("least developed countries",
"EU institutions") run under 40; the heading and departures echo the label."""


class EntityGroup(StrictModel):
    """A named group whose members fill one entity role, as the request named it
    ("EU member states"). The members themselves are the role's dcids in
    `BaseIdentity.entities`. This records only that they were asked for as one
    group, so the heading can say "the 27 EU member states" after any edit that
    keeps the role's members. A group the graph holds no node for, read from the
    membership registry ("Asia (DAC)"), names the registry's id for it in
    `group_id`, so a resubmit restores it and the total its dataset publishes for it.
    A group whose `ref` is a continent or subregion
    that the membership policy holds or omits (`MembershipPolicy.holds_region`)
    reads as its countries ("the 54 countries in Africa"), whatever its graph
    type. `entity_type` is informational, and the reading keys on `ref`. When
    the role binds the total its dataset publishes
    for the group ("Africa (DAC)") in place of the members, `total` names that
    party and the role's one dcid is it. A role read as the whole population of
    its side, the source's total of every party there ("Official Donors, Total")
    or a union of such totals ("all donors including private"), states the
    kinds of party that population holds in `whole_of`; a resubmit derives it
    from the identity and the code index, whatever it posts. A Spec serves its groups on
    `ResolvedSpec.entity_groups`, and a resubmit posts them beside the identity
    (`ResubmitIdentityEntry.entity_groups`, `ResubmitCandidate.entity_groups`)."""

    ref: IdentityString | None = Field(
        default=None,
        description="the group's own graph node, when the graph holds one; null otherwise",
    )
    label: str = Field(
        min_length=1,
        max_length=_ENTITY_GROUP_LABEL_MAX_LENGTH,
        description="the group's name as the request gave it, plain text",
    )
    entity_type: IdentityString | None = Field(
        default=None,
        description=(
            "the graph type of the group's own node, e.g. Continent, UNGeoRegion, "
            "ContinentalUnion; null when the graph holds no node for the group"
        ),
    )
    total: IdentityString | None = Field(
        default=None,
        description=(
            "the party the dataset publishes as the group's total, which the role binds "
            "in place of the members (DACAfrica for Africa); null when the role binds the "
            "members"
        ),
    )
    group_id: IdentityString | None = Field(
        default=None,
        description=(
            "the membership registry's id for a group the graph holds no node for, e.g. "
            "groups/OECD.DAC.Region.Asia; null when ref is set or the group "
            "is no registry group"
        ),
    )

    whole_of: list[PartyKind] | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            'the kinds of party the role\'s whole population holds, e.g. ["party"] for '
            "Official Donors, Total, which holds every official donor and no private one; "
            "absent for any other group"
        ),
    )

    @model_validator(mode="after")
    def _one_id(self) -> EntityGroup:
        if self.ref is not None and self.group_id is not None:
            raise ValueError("a group carries ref or group_id, not both")
        return self
