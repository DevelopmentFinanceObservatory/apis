"""Whole totals: the published total of every party on one side of a flow.

The code index marks a total as whole in a dataset where it holds every party of some kinds
on its one axis (`code_index.types.WholeMarker`): "Official Donors, Total"
(`DACOfficialDonors`) holds every `party` donor in DAC2a and DAC2b, and "All Recipients,
Total" (`DACAllRecipients`) every `party` and `residual` recipient in DAC2a from 1984. It
also names unions of whole totals read as one population (`code_index.types.WholeTotalSet`):
"all donors including private" is the official donors' total and the private donors'.

A side reads as its whole population (`whole_reading`) where it binds exactly one whole total
whose marker holds the `party` kind, or exactly the members of one whole total set, in every
dataset the answer is read from, within the marker's years. A total of a narrower population,
"Private Donors, Total" alone, is a party the request names and no whole population. Search
binds such a reading on a side the part totals (`place_roles.whole_total_candidates`), reads
a side a text names as a whole population so where it holds (`place_roles.bind_place_roles`),
and reads that side as unnamed on a dataset that publishes no total of a population covering
it (`covers_unpublished`). A Spec carries the reading on `resolved.entity_groups`
(`EntityGroup.whole_of`), on a first answer and on a resubmit alike. An answer that lists a
side member by member where its dataset publishes such a total states the total beside the
members in a `whole_total_available` warning (`whole_total_warnings`).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from code_index.shared import get_index
from code_index.types import PartyKind, PartyRow, WholeMarker, WholeTotalSet
from dfo_platform.roles import role_key_axis
from search.engine.extract import DateRequest
from search.engine.region.codes import WHOLE_TOTAL_AVAILABLE
from search.engine.region.coverage_series import series_held
from search.engine.retrieval.bound_values import identity_dataset
from search.engine.spec_build import NamedGroup
from search.heading.departure_fact import DepartureFact
from search.models import IdentityWindow, RoleAxis, Spec, TimeWindow, Warning, role_ref
from search.render.departures import (
    WHOLE_MEMBER_NOUNS,
    whole_population_is_official,
    whole_population_text,
    whole_residual_text,
)

if TYPE_CHECKING:
    from series_coverage.index import CoverageIndex

# The kind every whole population holds: a side read whole holds its ordinary parties.
_POPULATION_KIND: PartyKind = "party"


@dataclass(frozen=True, slots=True)
class WholeReading:
    """One side of a flow read as its whole population.

    `members` are the parties the side binds: one whole total, or the members of one whole
    total set. `total` is the one whole total, `None` for a set. `label` names the population:
    the total's graph label, or the set's phrase. `whole_of` is the kinds of party the
    population holds, sorted: the total's marker's, or the union of the set members'.
    """

    axis: RoleAxis
    members: tuple[str, ...]
    total: str | None
    label: str
    whole_of: tuple[PartyKind, ...]

    def named_group(self) -> NamedGroup:
        """The group a Spec carries for the side (`spec_build.role_entity_groups`): a single
        total names itself as the role's total and holds no members of its own, so a role
        binding it is no named set of members; a set's members are the role's entities."""
        return NamedGroup(
            label=self.label,
            dcid=None,
            member_dcids=frozenset() if self.total is not None else frozenset(self.members),
            total=self.total,
            whole_of=self.whole_of,
        )


def requested_years(date_request: DateRequest | None) -> tuple[int, int] | None:
    """The first and last year a part asks for, `None` unless its window states both."""
    window = None if date_request is None else date_request.window
    if window is None or window.start_year is None or window.end_year is None:
        return None
    return window.start_year, window.end_year


def identity_years(window: IdentityWindow) -> tuple[int, int] | None:
    """The first and last year an identity's window asks for, `None` unless it is a range
    stating both."""
    if window.kind != "range" or window.start_year is None or window.end_year is None:
        return None
    return window.start_year, window.end_year


def _within(marker: WholeMarker, years: tuple[int, int] | None) -> bool:
    """Whether `marker` holds in every year of `years`. A marker with no window holds in
    every year. A windowed marker holds only for a request stating both ends inside it: an
    open end, `latest` or no period at all can reach a year the composition check behind the
    window never read."""
    if marker.years is None:
        return True
    if years is None:
        return False
    first, last = marker.years
    return first <= years[0] and years[1] <= last


def _markers(party: PartyRow, datasets: Sequence[str]) -> list[WholeMarker] | None:
    """`party`'s marker in each of `datasets`, `None` when one of them marks it not."""
    markers = [marker for dataset in datasets if (marker := party.marker(dataset)) is not None]
    return markers if len(markers) == len(datasets) else None


def _whole_set_of(members: frozenset[str]) -> WholeTotalSet | None:
    """The whole total set whose members are exactly `members`, or `None`."""
    return next((s for s in get_index().whole_total_sets if frozenset(s.members) == members), None)


def whole_reading(
    dcids: Collection[str], *, datasets: Sequence[str], years: tuple[int, int] | None
) -> WholeReading | None:
    """The whole population `dcids` stand for on their flow axis when read from every one of
    `datasets` over `years` (`requested_years`), or `None`.

    One dcid reads whole when every dataset marks it within `years` and the markers agree on
    what it holds, which includes the `party` kind. Several read whole when they are exactly
    the members of one whole total set and each member reads so in every dataset, its kinds
    then the union of theirs. No dataset reads nothing whole."""
    index = get_index()
    members = frozenset(dcids)
    if not datasets or not members:
        return None
    if len(members) == 1:
        (dcid,) = members
        whole_set = None
        parties = [index.parties.get(dcid)]
    else:
        whole_set = _whole_set_of(members)
        if whole_set is None:
            return None
        parties = [index.parties.get(dcid) for dcid in whole_set.members]
    held: set[PartyKind] = set()
    for party in parties:
        if party is None or (markers := _markers(party, datasets)) is None:
            return None
        if len({marker.holds for marker in markers}) != 1:
            return None
        if not all(_within(marker, years) for marker in markers):
            return None
        held.update(markers[0].holds)
    if _POPULATION_KIND not in held:
        return None
    first = parties[0]
    assert first is not None
    if whole_set is not None:
        return WholeReading(
            axis=whole_set.axis,
            members=whole_set.members,
            total=None,
            label=whole_set.phrase,
            whole_of=tuple(sorted(held)),
        )
    return WholeReading(
        axis=first.axes[0],
        members=(first.dcid,),
        total=first.dcid,
        label=first.label or first.dcid,
        whole_of=tuple(sorted(held)),
    )


def covers_unpublished(dcids: Collection[str], *, datasets: Sequence[str]) -> bool:
    """Whether the whole population `dcids` names reads, on a shape answered from
    `datasets`, as its side left unnamed: none of `datasets` marks any of its parties, so
    none publishes its total, those datasets hold some party on its side, and every one,
    totals aside, is of a kind its markers elsewhere say it holds. The CRS publishes no
    total of its recipients and holds `party` and `residual` ones, so "all recipients"
    (`DACAllRecipients`, `party` and `residual`) reads as every recipient it holds. It
    holds private donors beside official ones, so "official donors" (`DACOfficialDonors`,
    `party` alone) never reads as every donor it holds; DAC1 holds official donors alone,
    so there it does."""
    index = get_index()
    rows = [index.parties.get(dcid) for dcid in dcids]
    if not datasets or not rows or any(row is None for row in rows):
        return False
    parties = cast(list[PartyRow], rows)
    if any(row.marker(dataset) is not None for row in parties for dataset in datasets):
        return False
    holds = {kind for row in parties for marker in row.whole for kind in marker.holds}
    kinds = side_kinds(parties[0].axes[0], datasets=datasets)
    return bool(kinds) and set(kinds) <= holds


def side_kinds(axis: RoleAxis, *, datasets: Sequence[str]) -> tuple[PartyKind, ...]:
    """The kinds of party, totals aside, that `datasets` hold on `axis`, sorted."""
    return tuple(
        sorted(
            {
                party.kind
                for party in get_index().parties.values()
                if party.kind != "total"
                and axis in party.axes
                and any(dataset in party.datasets for dataset in datasets)
            }
        )
    )


def whole_total_unpublished_facts(
    spec: Spec, whole_named: Mapping[str, tuple[str, ...]]
) -> list[DepartureFact]:
    """`whole_total_unpublished` for each whole population a text of the part names
    (`whole_named`, each text with the dcids it names) whose side `spec`'s identity leaves
    unnamed because the dataset elected for it publishes no total of the population and
    holds no party outside it there (`covers_unpublished`). `asked` is the text, `shown`
    the dataset's name, `over` the side and `whole_of` the kinds of party the dataset holds
    there (`side_kinds`). A door that holds no words of the request (a resubmit) names no
    population, so it states none."""
    identity = spec.definition.identity
    dataset = identity_dataset(identity)
    if dataset is None:
        return []
    bound = {
        axis for role, dcids in identity.entities.items() if dcids and (axis := role_key_axis(role))
    }
    shown = next(
        (source.label for source in spec.resolved.trace.resolved_sources if source.ref == dataset),
        dataset,
    )
    parties = get_index().parties
    facts: list[DepartureFact] = []
    for text, dcids in whole_named.items():
        axis = parties[dcids[0]].axes[0]
        if axis in bound or not covers_unpublished(dcids, datasets=(dataset,)):
            continue
        facts.append(
            DepartureFact(
                code="whole_total_unpublished",
                binding_ref=role_ref(axis),
                asked=text,
                shown=shown,
                over=axis,
                whole_of=side_kinds(axis, datasets=(dataset,)),
            )
        )
    return facts


def named_whole(text: str, resolved: str | None) -> tuple[str, ...] | None:
    """The parties a text of the part names as a whole population, `None` when it names
    none: the members of the whole total set whose phrase it is, else the one party it
    resolved to (`resolved`) when some dataset marks that party as holding the `party`
    kind. "All donors" is a phrase of the official and private donors' set; "official
    donors" resolves to `DACOfficialDonors`; "private donors" resolves to
    `DACPrivateDonors`, which holds private donors alone, and names a party, not a
    population."""
    index = get_index()
    if (whole_set := index.whole_set(text)) is not None:
        return whole_set.members
    party = None if resolved is None else index.parties.get(resolved)
    if party is None or not any(_POPULATION_KIND in marker.holds for marker in party.whole):
        return None
    return (party.dcid,)


def named_whole_label(dcids: Collection[str]) -> str | None:
    """The label a side that reads `dcids` whole carries (`WholeReading.label`): the phrase
    of the whole total set they are the members of, else the one party's label. `None` when
    they are neither one party nor one set's members."""
    index = get_index()
    members = frozenset(dcids)
    if len(members) > 1:
        whole_set = _whole_set_of(members)
        return None if whole_set is None else whole_set.phrase
    party = index.parties.get(next(iter(members))) if members else None
    return None if party is None else party.label or party.dcid


def population_totals(axes: Collection[RoleAxis]) -> tuple[str, ...]:
    """The parties some dataset marks as holding the `party` kind on one of `axes`, by
    dcid: the totals a side the part totals can bind (`place_roles.totalled_axes`)."""
    if not axes:
        return ()
    return tuple(
        sorted(
            party.dcid
            for party in get_index().parties.values()
            if party.whole
            and party.axes[0] in axes
            and any(_POPULATION_KIND in marker.holds for marker in party.whole)
        )
    )


def whole_total_warnings(spec: Spec, *, coverage: CoverageIndex) -> list[Warning]:
    """`whole_total_available` for each flow side `spec` lists member by member, opened by
    `open_roles` or defaulted (`ResolvedSpec.role_axis_facts`), where the dataset elected
    for its identity (`bound_values.identity_dataset`) publishes the total of that side's
    whole population over the identity's requested years (`whole_reading`) and `coverage`
    holds a series of the Spec's own variables at that total beside the places its other
    side binds (`coverage_series.series_held`). Those variables are among the ones binding
    repair asks of the total (`place_roles.holds_whole_total`), so the warning names no
    total binding would not serve. The warning names the total in `params`, so a caller
    that sums the members sees the published figure first, and is stamped with `spec`'s
    id: it holds for that reading alone, in a candidate set as in a definite answer.

    An answer that reads the other side whole states none: the source publishes no figure
    between two whole totals ("Official Donors, Total" to "All Recipients, Total"), so no
    published total stands beside the members' sum."""
    identity = spec.definition.identity
    dataset = identity_dataset(identity)
    groups = spec.resolved.entity_groups or {}
    if dataset is None or any(group.whole_of for group in groups.values()):
        return []
    years = identity_years(identity.requested_window)
    variables = [member.ref.ref for member in spec.resolved.members]
    window = _identity_window(identity.requested_window)
    bound: dict[RoleAxis, Collection[str]] = {
        axis: dcids
        for role, dcids in identity.entities.items()
        if (axis := role_key_axis(role)) is not None
    }

    def served(reading: WholeReading) -> bool:
        held = series_held(
            coverage,
            variables=variables,
            bound={**bound, reading.axis: reading.members},
            window=window,
        )
        return held is True

    readings = [
        reading
        for fact in spec.resolved.role_axis_facts or ()
        for dcid in population_totals((fact.role,))
        if (reading := whole_reading((dcid,), datasets=(dataset,), years=years)) is not None
        and served(reading)
    ]
    return [
        Warning(
            code=WHOLE_TOTAL_AVAILABLE,
            severity="info",
            message=_available_message(reading, dataset=dataset),
            spec_id=spec.spec_id,
            params={
                "total": reading.members[0],
                "total_label": reading.label,
                "whole_of": ",".join(reading.whole_of),
            },
        )
        for reading in readings
    ]


def _identity_window(window: IdentityWindow) -> TimeWindow | None:
    """The years an identity's window asks for, `None` for every year or the latest."""
    if window.kind != "range":
        return None
    return TimeWindow(start_year=window.start_year, end_year=window.end_year)


def _available_message(reading: WholeReading, *, dataset: str) -> str:
    """The `whole_total_available` message for a side whose dataset publishes `reading`:
    what the total holds, and how a sum of the listed members differs from it. A donor
    total that holds no private donor says the listed donors include them where `dataset`
    publishes private donors on that side."""
    noun = WHOLE_MEMBER_NOUNS.get(reading.axis, "member")
    population = whole_population_text(reading.axis, reading.whole_of)
    holds = whole_residual_text(reading.whole_of)
    private = whole_population_is_official(reading.axis, reading.whole_of) and any(
        party.kind == "private" and dataset in party.datasets and reading.axis in party.axes
        for party in get_index().parties.values()
    )
    differs = " includes private donors and" if private else ""
    return (
        f"The source publishes the total of {population} as {reading.label}{holds}; "
        f"a sum of the listed {noun}s{differs} differs from it."
    )
