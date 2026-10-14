"""The place set: the one value Search holds for each place a request names.

A request names a place by a text: a country, a continent or UN subregion, or a group the
resolver knows ("G7"). Each is one `PlaceSet`. A continent, subregion or group is a set with
members, and may also be a graph node of its own that holds observations, or a group a flow
dataset publishes a total party for. A country is a set
of one: a node with no members. `ResolveState.place_sets` holds the sets from place
resolution onward, keyed by the request's text.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

# What the request's text named: one place, a continent or UN subregion, or a resolver group.
PlaceSetKind = Literal["place", "region", "group"]

# Which of a set's two readings is asked for. `"members"` is every member as its own place,
# and `"default"` is the set's node where it has one.
PlaceReading = Literal["default", "members"]


@dataclass(frozen=True, slots=True)
class PlaceNode:
    """A place set's own graph node."""

    dcid: str
    graph_type: str


@dataclass(frozen=True, slots=True)
class PlaceSet:
    """One place a request names: its graph node, its members, and the reading asked of it.

    A set holds a node, members, or both. A place and a region always hold a node. A group
    holds members, and a node only when the resolver confirmed one for it.
    """

    # The request's text for the set, and its key in `ResolveState.place_sets`.
    name: str
    kind: PlaceSetKind
    node: PlaceNode | None
    # The graph's label for `node`, read for a group and a region. `None` for a place, and
    # when the set has no node or the graph gives it no label.
    label: str | None = None
    # The set's member dcids, in the order their source gave them: the membership policy's
    # for a region, the resolver's for a group. `None` when the set does not expand (a place,
    # or a region the membership policy does not hold).
    members: tuple[str, ...] | None = None
    # The graph type the members share. `None` when the set does not expand.
    member_type: str | None = None
    # The resolver's name for each member, by dcid. A group's only.
    member_names: Mapping[str, str] = field(default_factory=dict)
    reading: PlaceReading = "default"
    # The party a flow dataset publishes as the set's total ("Africa (DAC)" for Africa), as
    # the code index declares it (`CodeIndex.totals_by_group`). `None` for a set no total
    # holds. A flow role can bind it in place of the members (`place_roles.set_choices`).
    total: str | None = None
    # A group's id in the membership registry (`entity_resolution.Group.dcid`), such as
    # `groups/OECD.DAC.Region.Asia`. `None` for a place and a region.
    group_id: str | None = None

    @property
    def expands(self) -> bool:
        """Whether the set has a member list, which may be empty."""
        return self.members is not None

    def dcids(self, reading: PlaceReading | None = None) -> tuple[str, ...]:
        """The set's places under `reading`, or under its own reading when none is given.

        `"members"` is the members, and the node alone for a set that does not expand.
        `"default"` is the node, and the members for a set that has no node.
        """
        node = () if self.node is None else (self.node.dcid,)
        members = self.members
        if (reading or self.reading) == "members":
            return node if members is None else members
        return node or members or ()
