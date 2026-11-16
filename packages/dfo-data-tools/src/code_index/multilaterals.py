"""Which parties are multilateral organisations, read from the code index, so Search (donors)
and the mint (DAC2a destinations) classify a party the same way.

A multilateral is an official party the graph types `Organization` (`PartyRow.types`). The
foundations, which the graph types `Organization` too, are private providers
(`PartyRow.party_class`) and so are no multilateral. The EU institutions count, since they
spend their member states' contributions
(`code_index.scope_conventions.REDISTRIBUTED_PROVIDERS`).
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Final, NamedTuple

from code_index.types import PartyClass, PartyKind, PartyRow, RoleAxis

COUNTRY_PREFIX: Final[str] = "country/"
"""The dcid prefix of a country node."""

ORGANIZATION_TYPE: Final[str] = "Organization"
"""The graph type of an organisation, multilateral or private (`PartyRow.types`)."""


class PartyFacts(NamedTuple):
    """What one party is where some datasets publish it on one axis: its kind
    (`PartyRow.kind_over`), its class (`PartyRow.class_of`), and whether it is a multilateral,
    each `None` where those datasets give it different kinds."""

    kind: PartyKind | None
    party_class: PartyClass | None
    multilateral: bool | None


def party_facts(row: PartyRow, *, axis: RoleAxis, datasets: Collection[str] = ()) -> PartyFacts:
    """The facts of `row` where `datasets` publish it on `axis`. A multilateral is a party the
    graph types `Organization` that is an official party there; a total, such as the World Bank
    Group beside its members in DAC2a, is none."""
    kind = row.kind_over(axis, datasets)
    if kind is None:
        return PartyFacts(None, None, None)
    party_class = row.class_of(kind)
    multilateral = kind == "party" and party_class == "official" and ORGANIZATION_TYPE in row.types
    return PartyFacts(kind, party_class, multilateral)


def is_multilateral(row: PartyRow | None, *, datasets: Collection[str] = ()) -> bool:
    """Whether `row` is a multilateral where `datasets` publish it as a donor, on the origin
    axis (`party_facts`)."""
    return (
        row is not None and party_facts(row, axis="origin", datasets=datasets).multilateral is True
    )
