"""Which donors are multilateral organisations, whose outflows spend the core contributions
their funders made to them.

A donor's total that counts its core contributions to the multilaterals (DAC1's total ODA)
already holds what a multilateral then spends from them, so a caller summing the two counts
the money twice. Search states which parties are multilaterals (`ResolvedEntity.multilateral`)
and whether a donor population can hold one (`EntityGroup.holds_multilaterals`,
`RoleAxisFact.holds_multilaterals`).

A multilateral is an official party the graph types `Organization`: the organisation records
the entity resolver exports from the graph (`entity_resolution.pack.organisation_dcids`), less
the private providers, which the code index classes `private` (`PartyRow.party_class`). The
EU institutions count, since they spend their member states' contributions
(`code_index.scope_conventions.REDISTRIBUTED_PROVIDERS`).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping

from code_index.query import CodeIndex
from code_index.shared import per_index
from code_index.types import PartyRow, RoleAxis
from entity_resolution.pack import organisation_dcids


def is_multilateral(row: PartyRow | None, *, datasets: Collection[str] = ()) -> bool:
    """Whether `row` is a multilateral: an organisation that is an official party where
    `datasets` publish it (`PartyRow.kind_over`, `PartyRow.class_of`). A total, such as the
    World Bank Group beside its members in DAC2a, is none."""
    if row is None or row.dcid not in organisation_dcids():
        return False
    kind = row.kind_over(datasets)
    return kind == "party" and row.class_of(kind) == "official"


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
