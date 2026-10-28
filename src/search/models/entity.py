from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .base import StrictModel
from .grammar import GraphRef
from .identity import IdentityString
from .vocab import Confidence, EntityRoleKey, PartyClass, RoleAxis

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


def _is_none(value: object) -> bool:
    return value is None


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
    multilateral: bool | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "true for an origin entity that is a multilateral organisation, an official "
            "organisation such as IDA or the EU institutions, whose outflows spend the core "
            "contributions its funders made to it, so a donor total counting those "
            "contributions already holds them; absent for every other entity"
        ),
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
    keeps the role's members. A group the graph holds no node for, read from the
    membership registry ("Asia (DAC)"), names the registry's id for it in
    `group_id`, so a resubmit restores it and the total its dataset publishes for it.
    A group whose `ref` is a continent or subregion
    that the membership policy holds or omits (`MembershipPolicy.holds_region`)
    reads as its countries ("the 54 countries in Africa"), whatever its graph
    type. `entity_type` is informational, and the reading keys on `ref`. When
    the role binds the total its dataset publishes
    for the group ("Africa (DAC)") in place of the members, `total` names that
    party and the role's one dcid is it, and `members` lists the group's members as the
    membership registry holds them, so a caller can tell whether a party sits inside the
    total. A role read as the whole population of
    its side, the source's total of every party there ("Official Donors, Total")
    or a union of such totals ("all donors including private"), states the
    classes of party that population covers in `whole_of`, and the classes the answer's
    dataset publishes on that side in `side_classes`; a resubmit derives both from the
    identity and the code index, whatever it posts. A role read as the group's members states
    the residuals the group's total counts that the members leave out in `left_out`, and
    whether the source publishes its own figure for the group in `sum_differs`
    (`spec_build.with_members_left_out`). A Spec serves its groups on
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

    members: list[IdentityString] | None = Field(
        default=None,
        max_length=1024,
        exclude_if=_is_none,
        description=(
            "for a role that binds the group's published total (total), the group's members "
            "as the membership registry lists them, sorted, e.g. the 32 DAC member countries "
            "for DAC countries, total; the residuals a regional total counts beside them are "
            "not listed. Absent when the role binds the members or the registry lists none "
            "for the group. Derived on every answer; a posted value is ignored"
        ),
    )
    whole_of: list[PartyClass] | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "the classes of party the role's whole population covers, sorted: official, "
            'private or residual, e.g. ["official"] for Official Donors, Total, which holds '
            "every official donor and no private one; absent for any other group"
        ),
    )
    holds_multilaterals: bool | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "for a donor role that binds a published total (total) or reads as its whole "
            "population (whole_of), whether that population holds a multilateral organisation "
            "(see ResolvedEntity.multilateral): false for DAC countries, total, whose members "
            "are countries, true for Official Donors, Total in DAC2a. Absent for any other "
            "group and for a total whose members the registry does not list. Derived on every "
            "answer; a posted value is ignored"
        ),
    )
    side_classes: list[PartyClass] | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "for a role read as its whole population, the classes of party, totals aside, the "
            'answer\'s dataset publishes on that side, sorted, e.g. ["official", "private", '
            '"residual"] for donors in DAC2a; a class here and not in whole_of, such as '
            '"private", is outside the total. Absent for any other group. Derived from the '
            "identity on every answer; a posted value is ignored"
        ),
    )

    left_out: list[str] | None = Field(
        default=None,
        max_length=64,
        exclude_if=_is_none,
        description=(
            "for a group the role reads as its members, the labels of the residuals the "
            "group's published total counts beside the members that the answer leaves out "
            "and its dataset publishes on the role's side, in the total's order, e.g. "
            '["Africa, unallocated by country (DAC)"] for Africa read as its countries. Absent '
            "when the members leave nothing out, for a group the role reads as its total or "
            "whole population, and on a role outside a flow. Derived on every answer; a posted "
            "value is ignored"
        ),
    )
    sum_differs: bool | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "for a group the role reads as its members, true when the dataset publishes its "
            "own figure for the group's total and that figure is not the members' sum, so it "
            "can differ from one built from them. Absent otherwise. Derived on every answer; "
            "a posted value is ignored"
        ),
    )

    @model_validator(mode="after")
    def _one_id(self) -> EntityGroup:
        if self.ref is not None and self.group_id is not None:
            raise ValueError("a group carries ref or group_id, not both")
        return self
