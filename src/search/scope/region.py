"""The region axis: whether a part that sums over a recipient region counts the region's
residual buckets ("Africa, unallocated by country") beside its member countries.

A region read as one sum binds its members and the residuals its total counts beside them
(`PartyRow.residuals`), where the served dataset publishes them on the recipient axis, so
the members and residuals together equal the region's total. A ranking, or any other
operation that needs every member's own value, reads the members alone, and so does a
region the request names for its members ("African countries"). The parties a residual
row names are kept out of every member count by their party kind (`is_residual`).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence

from code_index.query import CodeIndex
from code_index.types import PartyRow
from search.models import (
    BaseIdentity,
    EntitiesBinding,
    Operation,
    Option,
    OptionAvailability,
    RoleAxis,
)
from search.scope.state import AxisRecord
from search.scope.tables import AxisTable, ScopeFacts
from search.scope.vocabulary import REGION_MEMBERS_AND_RESIDUALS, REGION_MEMBERS_ONLY

REGION_RULE = "region_residuals"
"""The declared rule the region axis applies: a summed region counts its residuals."""

REGION_ROLE = "destination"
"""The role key a region on the recipient side binds under."""

_AXIS: RoleAxis = "destination"


def is_residual(dcid: str, index: CodeIndex) -> bool:
    """Whether `dcid` is a party the code index lists as a residual in every dataset that
    publishes it (`PartyRow.kinds_over`), such as "Africa, unallocated by country"."""
    party = index.parties.get(dcid)
    return party is not None and party.kinds_over() == {"residual"}


def without_residuals(dcids: Iterable[str], index: CodeIndex) -> list[str]:
    """`dcids` in order, less every residual (`is_residual`)."""
    return [dcid for dcid in dcids if not is_residual(dcid, index)]


def published_residuals(
    total: PartyRow, *, datasets: Collection[str], index: CodeIndex
) -> tuple[str, ...]:
    """The residuals `total` counts beside its members that one of `datasets` publishes
    on the recipient axis, sorted. Empty for no `datasets`, and for a total that is a
    published figure of its own (`PartyRow.sums_members` false), whose residuals the index
    leaves empty."""
    served = set(datasets)
    return tuple(
        sorted(
            dcid
            for dcid in total.residuals
            if (party := index.parties.get(dcid)) is not None
            and _AXIS in party.axes
            and served.intersection(party.datasets)
        )
    )


def sums_recipients(operations: Sequence[Operation]) -> bool:
    """Whether a part with `operations` reads its recipient region as one sum: no operation
    other than `aggregate` runs over the recipients (`destination`) or over the places
    (`place`). A rank, share or growth over them needs every member's own value."""
    return all(
        operation.verb == "aggregate"
        for operation in operations
        if operation.over in (_AXIS, "place")
    )


def region_total(
    held: Sequence[str], *, group_refs: Sequence[str], index: CodeIndex
) -> PartyRow | None:
    """The total that sums the region a role reads: the total holding the role's group
    (`CodeIndex.totals_by_group`, by the group's node, else its registry id), else the
    total counting the most of the residuals the role holds, the first by dcid among
    equals. None for a role reading no summed region."""
    for ref in group_refs:
        total = index.totals_by_group.get(ref)
        if total is not None and total.sums_members and total.residuals:
            return total
    held_set = set(held)
    counted = [
        (len(held_set.intersection(total.residuals)), total)
        for total in index.totals_by_group.values()
        if total.sums_members
    ]
    best = min(
        ((count, total) for count, total in counted if count),
        key=lambda pair: (-pair[0], pair[1].dcid),
        default=None,
    )
    return None if best is None else best[1]


def region_rule(
    base: BaseIdentity, record: AxisRecord | None, facts: ScopeFacts
) -> AxisTable | None:
    """The region axis of a part whose identity is `base`, or None where the part reads no
    region its dataset counts residuals for.

    The axis is present when the recipient role reads the members of a region whose total
    sums them with residuals (`region_total`), and binds neither the total itself nor
    nothing. `region:members_and_residuals` adds the residuals the role holds, else those
    the served datasets publish there (`published_residuals`); `region:members_only` adds
    none. A part that does not sum the region (`sums_recipients`) offers the residuals as
    unavailable, `residuals_need_summed_region`, and takes the members alone. `record`,
    the binder's, adds nothing the identity and `facts.part` do not state, so a
    dataset_resubmit, where no binder runs, builds the same table."""
    held = base.entities.get(REGION_ROLE)
    if not held:
        return None
    index = facts.index
    part = facts.part
    group = part.entity_groups.get(REGION_ROLE)
    group_refs = [] if group is None else [ref for ref in (group.ref, group.group_id) if ref]
    total = region_total(held, group_refs=group_refs, index=index)
    if total is None or total.dcid in held:
        return None
    bound = [dcid for dcid in held if dcid in total.residuals]
    residuals = tuple(sorted(bound)) or published_residuals(
        total, datasets=part.datasets, index=index
    )
    if not residuals or not without_residuals(held, index):
        return None
    summed = sums_recipients(part.operations)
    with_residuals = Option(
        option_id=REGION_MEMBERS_AND_RESIDUALS,
        label="the member countries and the amounts not allocated to any one of them",
        binding=EntitiesBinding(role=REGION_ROLE, add=list(residuals)),
        availability=(
            OptionAvailability(status="available")
            if summed
            else OptionAvailability(status="unavailable", code="residuals_need_summed_region")
        ),
    )
    members_only = Option(
        option_id=REGION_MEMBERS_ONLY,
        label="the member countries alone",
        binding=EntitiesBinding(role=REGION_ROLE, add=[]),
        availability=OptionAvailability(status="available"),
    )
    return AxisTable(
        axis="region",
        rule=REGION_RULE,
        options=(with_residuals, members_only),
        preference=(
            (REGION_MEMBERS_AND_RESIDUALS, REGION_MEMBERS_ONLY)
            if summed
            else (REGION_MEMBERS_ONLY,)
        ),
        held=REGION_MEMBERS_AND_RESIDUALS if bound else REGION_MEMBERS_ONLY,
    )
