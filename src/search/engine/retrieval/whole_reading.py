"""Whether a side of a flow reads as its whole population, from the code index alone.

The code index marks a total as whole in a dataset where it holds every party of some classes
on its one axis (`code_index.types.WholeMarker`), over the marker's years. `whole_reading`
reads one total, or the members of one whole total set, against those markers, and
`identity_whole_groups` reads each role of an identity over its own dataset and window, and
`whole_groups_over_identity` restricts that reading to the roles a door reads: a first answer
the roles its request named or was offered a whole population on
(`spec_build.answer_identity`), a resubmit the roles it posts a whole group or no group on
(`resubmit_request.resubmit_named_groups`). The candidates, departures and warnings built on
it live in `region.whole_totals`.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass, replace

from code_index.shared import get_index
from code_index.types import POPULATION_CLASS, PartyClass, PartyRow, WholeMarker, WholeTotalSet
from dfo_platform.roles import role_key_axis
from search.engine.extract import TrailingSpan
from search.engine.named_group import NamedGroup
from search.engine.retrieval.bound_values import identity_dataset
from search.models import BaseIdentity, IdentityWindow, RoleAxis


@dataclass(frozen=True, slots=True)
class WholeReading:
    """One side of a flow read as its whole population.

    `members` are the parties the side binds: one whole total, or the members of one whole
    total set. `total` is the one whole total, `None` for a set. `label` names the population:
    the total's graph label, or the set's phrase. `whole_of` is the classes of party the
    population covers, sorted: the total's marker's, or the union of the set members'. A
    containing total (`region.whole_totals.containing_reading`) covers no class: `whole_of`
    is empty and `group_id` names the membership registry's group the total holds.
    """

    axis: RoleAxis
    members: tuple[str, ...]
    total: str | None
    label: str
    whole_of: tuple[PartyClass, ...]
    group_id: str | None = None

    def named_group(self) -> NamedGroup:
        """The group a Spec carries for the side (`spec_build.role_entity_groups`): a single
        total names itself as the role's total and holds no members of its own, so a role
        binding it is no named set of members; a set's members are the role's entities."""
        return NamedGroup(
            label=self.label,
            dcid=None,
            member_dcids=frozenset() if self.total is not None else frozenset(self.members),
            total=self.total,
            group_id=self.group_id,
            whole_of=self.whole_of,
        )


@dataclass(frozen=True, slots=True)
class YearsAsked:
    """The years a whole-total marker must hold over, `first` to `last`. `trailing` is the
    span counted back from now ("the last N years") whose years are those its newest year may
    lie in (`extract.DateRequest.judged`), and None for a plain range. `first_read` is the
    first year such a span's read lists (`scope_window.first_read_year`), where the range it
    settles on starts at the latest."""

    first: int
    last: int
    trailing: TrailingSpan | None = None
    first_read: int | None = None


def identity_years(window: IdentityWindow) -> YearsAsked | None:
    """The years an identity's window asks for, `None` unless it is a range stating both."""
    if window.kind != "range" or window.start_year is None or window.end_year is None:
        return None
    return YearsAsked(first=window.start_year, last=window.end_year)


def _within(marker: WholeMarker, years: YearsAsked | None) -> bool:
    """Whether `marker` holds in every year of `years`. A marker with no window holds in
    every year. A windowed marker holds only for a request stating both ends inside it: an
    open end, `latest` or no period at all can reach a year the composition check behind the
    window never read.

    A span counted back from now settles its years only once the answer's series are read
    (`region.trailing.settle_trailing`), after the reading has chosen what binds, so its
    check is provisional: the marker must hold over the N years ending at the earlier of the
    span's last year and the marker's own, starting no earlier than the first year the read
    lists (`YearsAsked.first_read`), the range the answer settles on when its data reach the
    dataset's last published year (`extract.TrailingSpan.settled`, the rule the answer
    settles by), and that year must lie in the years the span's newest year may lie in. The Spec the answer builds then reads its
    whole populations again over the range it settled on (`identity_whole_groups`), as a
    resubmit of it does, so the provisional check decides only whether a side the request
    leaves unnamed offers the total."""
    if marker.years is None:
        return True
    if years is None:
        return False
    first, last = marker.years
    if years.trailing is None:
        return first <= years.first and years.last <= last
    newest = min(years.last, last)
    settled = years.trailing.settled(newest, floor=years.first_read or first)
    assert settled.start_year is not None
    return first <= settled.start_year and years.first <= newest


def _markers(party: PartyRow, datasets: Sequence[str]) -> list[WholeMarker] | None:
    """`party`'s marker in each of `datasets`, `None` when one of them marks it not."""
    markers = [marker for dataset in datasets if (marker := party.marker(dataset)) is not None]
    return markers if len(markers) == len(datasets) else None


def whole_total_set_of(members: frozenset[str]) -> WholeTotalSet | None:
    """The whole total set whose members are exactly `members`, or `None`."""
    return next((s for s in get_index().whole_total_sets if frozenset(s.members) == members), None)


def whole_reading(
    dcids: Collection[str], *, datasets: Sequence[str], years: YearsAsked | None
) -> WholeReading | None:
    """The whole population `dcids` stand for on their flow axis when read from every one of
    `datasets` over `years` (`region.whole_totals.requested_years`), or `None`.

    One dcid reads whole when every dataset marks it within `years` and the markers agree on
    what it covers, which includes the `official` class. Several read whole when they are
    exactly the members of one whole total set and each member reads so in every dataset, its
    classes then the union of theirs. No dataset reads nothing whole."""
    return _reading(dcids, datasets=datasets, within=lambda marker: _within(marker, years))


def whole_population(dcids: Collection[str], *, datasets: Sequence[str]) -> WholeReading | None:
    """The whole population `dcids` stand for in every one of `datasets` over whatever years
    their markers hold (`whole_reading` with no years check), or `None`: the reading of a side
    a request names as one whole population alone, which the Spec reads again over its own
    window (`spec_build.answer_identity`)."""
    return _reading(dcids, datasets=datasets, within=lambda _marker: True)


def _reading(
    dcids: Collection[str],
    *,
    datasets: Sequence[str],
    within: Callable[[WholeMarker], bool],
) -> WholeReading | None:
    """`whole_reading` with each marker's years judged by `within`."""
    index = get_index()
    members = frozenset(dcids)
    if not datasets or not members:
        return None
    if len(members) == 1:
        (dcid,) = members
        whole_set = None
        parties = [index.parties.get(dcid)]
    else:
        whole_set = whole_total_set_of(members)
        if whole_set is None:
            return None
        parties = [index.parties.get(dcid) for dcid in whole_set.members]
    covered: set[PartyClass] = set()
    for party in parties:
        if party is None or (markers := _markers(party, datasets)) is None:
            return None
        if len({marker.covers for marker in markers}) != 1:
            return None
        if not all(within(marker) for marker in markers):
            return None
        covered.update(markers[0].covers)
    if POPULATION_CLASS not in covered:
        return None
    first = parties[0]
    assert first is not None
    if whole_set is not None:
        return WholeReading(
            axis=whole_set.axis,
            members=whole_set.members,
            total=None,
            label=whole_set.phrase,
            whole_of=tuple(sorted(covered)),
        )
    return WholeReading(
        axis=first.sole_axis(),
        members=(first.dcid,),
        total=first.dcid,
        label=first.label or first.dcid,
        whole_of=tuple(sorted(covered)),
    )


def identity_whole_groups(identity: BaseIdentity) -> dict[str, NamedGroup]:
    """The group of each flow role whose entities the dataset elected for `identity`
    (`bound_values.identity_dataset`) reads as the whole population of the role's side over
    the identity's requested window (`whole_reading`), by role key: one whole total ("ODA to
    Ethiopia" binds `DACOfficialDonors`) or the members of one whole total set. Empty for an
    identity whose dataset the code index does not elect.

    Each door reads it on the roles it picks (`whole_groups_over_identity`): a first answer on
    the roles its request named or was offered a whole population on, and a resubmit on the
    roles it posts a whole group or no group on, so both label such a role alike, a span
    counted back from now included. An identity naming every member of a whole total set
    reads as that set on a resubmit that posts no group on the role, since the identity does
    not carry the request's words:
    "official donors" beside "private donors" answers two named parties, and a resubmit of
    its identity reads "all donors including private"."""
    dataset = identity_dataset(identity)
    if dataset is None:
        return {}
    years = identity_years(identity.requested_window)
    groups: dict[str, NamedGroup] = {}
    for role, dcids in identity.entities.items():
        reading = whole_reading(dcids, datasets=(dataset,), years=years)
        if reading is not None and reading.axis == role_key_axis(role):
            groups[role] = reading.named_group()
    return groups


def unread_whole_group(group: NamedGroup, dcids: Collection[str]) -> NamedGroup | None:
    """A whole population (`group.whole_of`) whose identity does not read the role binding
    `dcids` whole, as both doors state it (`spec_build._restate_whole_groups`,
    `resubmit_request.resubmit_named_groups`): `None` where the group stands for one whole
    total, which the role then binds as a party, and the plain named set of the role's
    parties, labelled by the set's phrase, with no whole reading, where they are exactly the
    members of one whole total set ("all donors including private" in a year a member's
    marker does not hold). A resubmit reads a plain group it is posted back with, which
    names no graph node, through the same rule."""
    members = frozenset(dcids)
    if group.total is not None or (whole_set := whole_total_set_of(members)) is None:
        return None
    return replace(group, label=whole_set.phrase, member_dcids=members, whole_of=())


def whole_groups_over_identity(
    identity: BaseIdentity, roles: Iterable[str]
) -> dict[str, NamedGroup]:
    """The whole populations `identity_whole_groups` reads over `identity`, on `roles` only,
    by role key. Both doors read a role whole through it: a first answer on the roles its
    request named or was offered a whole population on (`spec_build.answer_identity`), and a
    resubmit on the roles it posts a whole group on or posts no group on
    (`resubmit_request.resubmit_named_groups`). A role outside `roles` gains no whole
    population."""
    asked = set(roles)
    return {role: group for role, group in identity_whole_groups(identity).items() if role in asked}
