"""The region axis: whether a part that sums over a recipient region counts the region's
residual buckets ("Africa, unallocated by country") beside its member countries.

A region read as one sum binds its members and the residuals its total counts beside them
(`PartyRow.residuals`), where the served dataset publishes them on the recipient axis, so
the members and residuals together equal the region's total. A ranking, or any other
operation that needs every member's own value, reads the members alone, and so does a
region the request names for its members ("African countries"). The parties a residual
row names are kept out of every member count by their party kind (`is_residual`).

A graph region the OECD counts with other members (`REGION_READINGS`: sub-Saharan Africa,
which the graph holds without Sudan) has two readings: the OECD's region, its DAC members
and residuals, which a summed part on a DAC dataset takes, and the graph region's own
member countries, which a ranking takes. The axis offers both, each label stating the
composition it counts.
"""

from __future__ import annotations

import functools
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass

from code_index.query import CodeIndex
from code_index.scope_conventions import (
    OPTION_LABELS,
    REGION_NAMES,
    REGION_READINGS,
    RegionReading,
)
from code_index.types import PartyRow
from entity_resolution.shared import shared_membership_policy
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
_REGION_NAMES = dict(REGION_NAMES)


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


_READINGS_BY_REGION = {row.region: row for row in REGION_READINGS}
_READINGS_BY_TOTAL = {row.total: row for row in REGION_READINGS}


def oecd_region_reading(region: str) -> RegionReading | None:
    """The declared OECD reading of graph region `region` (`REGION_READINGS`), None when
    the OECD counts it as the graph does or has no total for it."""
    return _READINGS_BY_REGION.get(region)


@functools.cache
def _policy_members(ref: str) -> frozenset[str]:
    """The members the membership policy holds for group or region `ref`, empty when it
    holds none. Cached by ref, since the policy is the process's
    (`shared_membership_policy`)."""
    policy = shared_membership_policy()
    if ref not in policy:
        return frozenset()
    held = policy.members(ref).members
    return frozenset(held) if isinstance(held, tuple) else frozenset()


def counts_oecd_region(total: PartyRow, *, datasets: Collection[str], index: CodeIndex) -> bool:
    """Whether one of `datasets` counts the OECD's region `total` sums: it publishes the
    total itself (DAC2a) or one of its residuals on the recipient axis (the CRS)."""
    return not set(datasets).isdisjoint(total.datasets) or bool(
        published_residuals(total, datasets=datasets, index=index)
    )


def region_total(
    held: Sequence[str], *, group_refs: Sequence[str], index: CodeIndex
) -> PartyRow | None:
    """The total that sums the region a role reads: the total holding the role's group
    (`CodeIndex.totals_by_group`, by the group's node, else its registry id), or the DAC
    total a graph region reads as (`oecd_region_reading`); else the total whose group's
    members, or whose graph region's (`REGION_READINGS`), are the members the role holds;
    else the total counting the most of the residuals the role holds, the one counting the
    fewest residuals among equals, so a subregion's total wins over its continent's, then
    the first by dcid. A role holding fewer than two members matches no region by its
    members. None for a role reading no summed region."""
    for ref in group_refs:
        row = oecd_region_reading(ref)
        total = index.parties.get(row.total) if row is not None else index.totals_by_group.get(ref)
        if total is not None and total.sums_members and total.residuals:
            return total
    members = frozenset(without_residuals(held, index))
    if len(members) > 1:
        for row in REGION_READINGS:
            total = index.parties.get(row.total)
            if (
                total is not None
                and total.totals is not None
                and members in (_policy_members(row.region), _policy_members(total.totals))
            ):
                return total
    held_set = set(held)
    counted = [
        (len(held_set.intersection(total.residuals)), total)
        for total in index.totals_by_group.values()
        if total.sums_members
    ]
    best = min(
        ((count, total) for count, total in counted if count),
        key=lambda pair: (-pair[0], len(pair[1].residuals), pair[1].dcid),
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
    none. A graph region the OECD counts with other members (`REGION_READINGS`), on a part a
    dataset publishing its DAC total serves, reads both ways (`_oecd_readings`). The option
    the identity holds comes first in the preference, so a part's derived default is the
    reading it resolved and a switch is the only thing that moves it. A part that does not
    sum the region (`sums_recipients`) offers the residuals as unavailable,
    `residuals_need_summed_region`, and takes the members alone. `record`, the binder's,
    adds nothing the identity and `facts.part` do not state, so a dataset_resubmit, where
    no binder runs, builds the same table. The labels name the region by its total's
    declared name (`REGION_NAMES`, "Africa", or the reading's, "sub-Saharan Africa"), else
    by the total's graph label, whatever groups the request posts, so a search and its
    switch name it alike."""
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
    members = without_residuals(held, index)
    if not members:
        return None
    reading = _READINGS_BY_TOTAL.get(total.dcid)
    if reading is not None:
        if not counts_oecd_region(total, datasets=part.datasets, index=index):
            return None
        readings = _oecd_readings(
            reading, total, residuals, held_members=frozenset(members), binds_residuals=bool(bound)
        )
    elif residuals:
        region = _REGION_NAMES.get(total.dcid) or total.label or total.dcid
        readings = _Readings(
            with_residuals=list(residuals),
            members_only=[],
            with_residuals_label=OPTION_LABELS.region_with_residuals.format(region=region),
            members_only_label=OPTION_LABELS.region_members_only.format(
                region_of=_possessive(region)
            ),
            held=REGION_MEMBERS_AND_RESIDUALS if bound else REGION_MEMBERS_ONLY,
        )
    else:
        return None
    if readings is None:
        return None
    summed = sums_recipients(part.operations)
    with_residuals = Option(
        option_id=REGION_MEMBERS_AND_RESIDUALS,
        label=readings.with_residuals_label,
        binding=EntitiesBinding(role=REGION_ROLE, add=readings.with_residuals),
        availability=(
            OptionAvailability(status="available")
            if summed
            else OptionAvailability(status="unavailable", code="residuals_need_summed_region")
        ),
    )
    members_only = Option(
        option_id=REGION_MEMBERS_ONLY,
        label=readings.members_only_label,
        binding=EntitiesBinding(role=REGION_ROLE, add=readings.members_only),
        availability=OptionAvailability(status="available"),
    )
    held_id = readings.held
    return AxisTable(
        axis="region",
        rule=REGION_RULE,
        options=(with_residuals, members_only),
        preference=(
            (
                held_id,
                *(o for o in (REGION_MEMBERS_AND_RESIDUALS, REGION_MEMBERS_ONLY) if o != held_id),
            )
            if summed
            else (REGION_MEMBERS_ONLY,)
        ),
        held=held_id,
    )


@dataclass(frozen=True, kw_only=True)
class _Readings:
    """What a region's two options add to the recipient role (`scope.edits`, which first
    removes what the other adds), their labels, and the option the identity holds."""

    with_residuals: list[str]
    members_only: list[str]
    with_residuals_label: str
    members_only_label: str
    held: str


def _oecd_readings(
    reading: RegionReading,
    total: PartyRow,
    residuals: Sequence[str],
    *,
    held_members: frozenset[str],
    binds_residuals: bool,
) -> _Readings | None:
    """The two readings of a graph region the OECD counts with other members: the OECD's
    region, the members of the group `total` sums with `residuals`, and the graph region's
    own members, labelled with the count each holds (`OPTION_LABELS.region_as_oecd_counts`,
    `region_place_members`). Each option adds what it holds that the other does not, the
    OECD's region its residuals and the countries only the OECD lists (Sudan for sub-Saharan
    Africa), the graph's the countries only the graph lists (Japan for Asia), so a switch
    removes the other's and adds its own (`scope.edits`). The identity holds the OECD's
    region when it binds a residual (`binds_residuals`) or the OECD's members where they
    differ from the graph's (`held_members`). None when the membership policy holds either
    group empty, or both readings bind the same places."""
    oecd = _policy_members(total.totals) if total.totals is not None else frozenset()
    place = _policy_members(reading.region)
    if not oecd or not place:
        return None
    as_oecd = [*sorted(oecd - place), *residuals]
    as_place = sorted(place - oecd)
    if not as_oecd and not as_place:
        return None
    holds_oecd = held_members == oecd and oecd != place
    return _Readings(
        with_residuals=as_oecd,
        members_only=as_place,
        with_residuals_label=OPTION_LABELS.region_as_oecd_counts.format(
            region=reading.name, count=len(oecd)
        ),
        members_only_label=OPTION_LABELS.region_place_members.format(
            region_of=_possessive(reading.name), count=len(place)
        ),
        held=(
            REGION_MEMBERS_AND_RESIDUALS if binds_residuals or holds_oecd else REGION_MEMBERS_ONLY
        ),
    )


def _possessive(name: str) -> str:
    """`name` as a possessive: "Africa's", "the Americas'"."""
    return f"{name}'" if name.endswith("s") else f"{name}'s"
