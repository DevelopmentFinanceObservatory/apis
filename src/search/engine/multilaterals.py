"""Which donors are multilateral organisations, whose outflows spend the core contributions
their funders made to them.

A donor's total that counts its core contributions to the multilaterals (DAC1's total ODA)
already holds what a multilateral then spends from them, so a caller summing the two counts
the money twice. Search states which parties are multilaterals (`ResolvedEntity.multilateral`)
and whether a donor population can hold one (`EntityGroup.holds_multilaterals`,
`RoleAxisFact.holds_multilaterals`). Each bound party's kind and class come with its marker
(`party_facts`).

A multilateral is an official party the graph types `Organization`: the organisation records
the entity resolver exports from the graph (`entity_resolution.pack.organisation_dcids`), less
the private providers, which the code index classes `private` (`PartyRow.party_class`). The
EU institutions count, since they spend their member states' contributions
(`code_index.scope_conventions.REDISTRIBUTED_PROVIDERS`).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from typing import NamedTuple

from code_index.query import CodeIndex
from code_index.shared import per_index
from code_index.types import PartyClass, PartyKind, PartyRow, RoleAxis
from entity_resolution.pack import organisation_dcids


class PartyFacts(NamedTuple):
    """What one party is where some datasets publish it: its kind (`PartyRow.kind_over`), its
    class (`PartyRow.class_of`), each `None` where the datasets give it different kinds, and
    whether it is a multilateral."""

    kind: PartyKind | None
    party_class: PartyClass | None
    multilateral: bool


def party_facts(row: PartyRow, *, datasets: Collection[str] = ()) -> PartyFacts:
    """The facts of `row` where `datasets` publish it. A multilateral is an organisation that
    is an official party there; a total, such as the World Bank Group beside its members in
    DAC2a, is none."""
    kind = row.kind_over(datasets)
    party_class = row.class_of(kind) if kind is not None else None
    multilateral = (
        kind == "party" and party_class == "official" and row.dcid in organisation_dcids()
    )
    return PartyFacts(kind, party_class, multilateral)


def is_multilateral(row: PartyRow | None, *, datasets: Collection[str] = ()) -> bool:
    """Whether `row` is a multilateral where `datasets` publish it (`party_facts`)."""
    return row is not None and party_facts(row, datasets=datasets).multilateral


def holds_multilaterals(
    dcids: Iterable[str], *, parties: Mapping[str, PartyRow], datasets: Collection[str] = ()
) -> bool:
    """Whether any of the `parties` rows `dcids` names is a multilateral (`is_multilateral`)."""
    return any(is_multilateral(parties.get(dcid), datasets=datasets) for dcid in dcids)


def side_holds_multilaterals(
    axis: RoleAxis, *, index: CodeIndex, datasets: Collection[str]
) -> bool:
    """Whether one of `datasets` publishes a multilateral on `axis`."""
    sides = _multilateral_sides(index)
    return any((axis, dataset) in sides for dataset in datasets)


@per_index
def _multilateral_sides(index: CodeIndex) -> frozenset[tuple[RoleAxis, str]]:
    """The `(axis, dataset)` sides on which a dataset publishes a multilateral."""
    return frozenset(
        (axis, dataset)
        for row in index.parties.values()
        for dataset in row.datasets
        if is_multilateral(row, datasets=(dataset,))
        for axis in row.axes
    )
