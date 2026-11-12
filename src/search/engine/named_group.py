"""A group a request named, and the entity dcids it stands for (`NamedGroup`)."""

from __future__ import annotations

from dataclasses import dataclass

from code_index.types import PartyClass


@dataclass(frozen=True)
class NamedGroup:
    """A group the request named ("EU member states") and the entity dcids it stands for.

    `label` is the group's name as the request gave it. `dcid` is its own graph node, when
    the graph holds one, and `entity_type` is that node's graph type. `named_alone` are the
    members the request also named on their own, each as the role key it fills and its
    dcid. `total` is the party a flow dataset publishes as the group's total
    (`PlaceSet.total`), which a role can bind in place of the members. `group_id` is the
    membership registry's id for a group the graph holds no node for (`EntityGroup.group_id`).
    `whole_of` is the classes of party a side read as its whole population covers
    (`retrieval.whole_reading.WholeReading`), empty for any other group.
    """

    label: str
    dcid: str | None
    member_dcids: frozenset[str]
    entity_type: str | None = None
    named_alone: frozenset[tuple[str, str]] = frozenset()
    total: str | None = None
    group_id: str | None = None
    whole_of: tuple[PartyClass, ...] = ()
