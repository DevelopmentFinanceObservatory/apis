"""Whole totals: the published total of every party on one side of a flow.

The code index marks a total as whole in a dataset where it holds every party of some classes
on its one axis (`code_index.types.WholeMarker.covers`): "Official Donors, Total"
(`DACOfficialDonors`) covers every `official` donor in DAC2a and DAC2b, and "All Recipients,
Total" (`DACAllRecipients`) every `official` and `residual` recipient in DAC2a from 1984. It
also names unions of whole totals read as one population (`code_index.types.WholeTotalSet`):
"all donors including private" is the official donors' total and the private donors'.

A side reads as its whole population (`retrieval.whole_reading.whole_reading`) where it binds
exactly one whole total whose marker covers the `official` class, or exactly the members of one
whole total set, in every dataset the answer is read from, within the marker's years. A total of a narrower population,
"Private Donors, Total" alone, is a party the request names and no whole population. Search
binds such a reading on a side the part totals (`place_roles.whole_total_candidates`), reads
a side a text names as a whole population so where it holds (`place_roles.bind_place_roles`),
and reads that side as unnamed on a dataset that publishes no total of a population covering
it (`covers_unpublished`). A Spec carries the reading on `resolved.entity_groups`
(`EntityGroup.whole_of`), read from its own identity (`whole_reading.whole_groups_over_identity`)
on the roles the request read whole, and on a resubmit also on the roles it posts no group
on, so a span counted back from now, whose years settle only once its series are read,
labels each side over the years it settled on. An answer that lists a side member by member
where its dataset publishes such a total states the total beside the members in a
`whole_total_available` warning (`whole_total_warnings`).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, cast

from code_index.declarations import CONTAINING_TOTALS
from code_index.shared import get_index
from code_index.types import (
    POPULATION_CLASS,
    PartyClass,
    PartyRow,
)
from dfo_platform.roles import role_key_axis
from entity_resolution.shared import shared_membership_policy
from search.engine.extract import DateRequest
from search.engine.region.codes import WHOLE_TOTAL_AVAILABLE
from search.engine.region.coverage_series import series_held
from search.engine.retrieval.bound_values import identity_dataset, served_members
from search.engine.retrieval.whole_reading import (
    WholeReading,
    YearsAsked,
    identity_years,
    whole_reading,
    whole_total_set_of,
)
from search.heading.departure_fact import DepartureFact
from search.models import (
    BaseIdentity,
    IdentityWindow,
    RoleAxis,
    Spec,
    TimeWindow,
    Warning,
    role_ref,
)
from search.render.departures import (
    WHOLE_MEMBER_NOUNS,
    private_outside,
    whole_population_text,
    whole_residual_text,
)
from search.scope_window import first_read_year

if TYPE_CHECKING:
    from series_coverage.index import CoverageIndex
    from series_coverage.schema import YearHorizon


def requested_years(date_request: DateRequest | None, *, horizon: YearHorizon) -> YearsAsked | None:
    """The years a part asks for, `None` unless its window states both ends. A span counted
    back from now asks for the years its newest year may lie in
    (`extract.DateRequest.judged`), short of a stale newest year, which departs, and its
    range starts no earlier than the first year a read over `horizon` lists
    (`scope_window.first_read_year`)."""
    judged = None if date_request is None else date_request.judged
    window = None if judged is None else judged.window
    if window is None or window.start_year is None or window.end_year is None:
        return None
    assert date_request is not None
    trailing = date_request.trailing
    return YearsAsked(
        first=window.start_year,
        last=window.end_year,
        trailing=trailing,
        first_read=None if trailing is None else first_read_year(horizon),
    )


def containing_reading(dcid: str, *, datasets: Sequence[str]) -> WholeReading | None:
    """The side `dcid` holds when every one of `datasets` serves it as the containing total of
    one flow axis (`code_index.declarations.CONTAINING_TOTALS`) and the code index publishes
    it on that axis in each of them (`PartyRow.publishes`), or `None`. The reading names
    the registry group the total holds (`PartyRow.totals`) where the membership policy lists
    that group, and covers no class."""
    row = get_index().parties.get(dcid)
    if row is None or not datasets:
        return None
    axes = {
        axis
        for (dataset, axis), held in CONTAINING_TOTALS.items()
        if dataset in datasets and held.total == dcid
    }
    if len(axes) != 1:
        return None
    (axis,) = axes
    if any(
        (held := CONTAINING_TOTALS.get((dataset, axis))) is None
        or held.total != dcid
        or not row.publishes(axis, (dataset,))
        for dataset in datasets
    ):
        return None
    membership = shared_membership_policy()
    return WholeReading(
        axis=axis,
        members=(dcid,),
        total=dcid,
        label=row.label or dcid,
        whole_of=(),
        group_id=row.totals if row.totals is not None and row.totals in membership else None,
    )


def containing_totals(axes: Collection[RoleAxis]) -> tuple[str, ...]:
    """The containing totals some dataset serves on one of `axes`, by dcid
    (`code_index.declarations.CONTAINING_TOTALS`)."""
    return tuple(
        sorted({held.total for (_, axis), held in CONTAINING_TOTALS.items() if axis in axes})
    )


_FLOW_AXES: tuple[RoleAxis, ...] = ("origin", "destination")


def with_containing_total(identity: BaseIdentity, *, coverage: CoverageIndex) -> BaseIdentity:
    """`identity` with the containing total of its dataset (`containing_reading`) bound on
    each flow side the identity leaves open, binding no entity there and not asking for every
    member (`BaseIdentity.open_roles`), where `coverage` holds a series of the variables the
    identity serves (`bound_values.served_members`) at the total beside its other places. A
    resubmit reads its posted identity so (`entry.dataset_resubmit`), as a fresh search binds
    the total on a side the part totals (`place_roles.whole_total_candidates`), so a no-op
    resubmit of a fresh answer answers the same Spec. Any other identity is returned as it
    is."""
    dataset = identity_dataset(identity)
    if dataset is None:
        return identity
    open_roles = set(identity.open_roles or ())
    # Entity keys keep the caller's spelling (`creditor`), so a side is read by its axis.
    bound: dict[RoleAxis, Collection[str]] = {
        axis: dcids
        for role, dcids in identity.entities.items()
        if (axis := role_key_axis(role)) is not None
    }
    unbound = {axis for axis in _FLOW_AXES if axis not in bound and axis not in open_roles}
    totals = {
        reading.axis: dcid
        for dcid in containing_totals(unbound)
        if (reading := containing_reading(dcid, datasets=(dataset,))) is not None
        and reading.axis in unbound
    }
    if not totals:
        return identity
    variables = served_members(identity)
    if not variables:
        return identity
    window = _identity_window(identity.requested_window)
    entities = dict(identity.entities)
    for axis, total in totals.items():
        if series_held(
            coverage, variables=variables, bound={**bound, axis: [total]}, window=window
        ):
            entities[axis] = [total]
    if entities == identity.entities:
        return identity
    return identity.model_copy(update={"entities": dict(sorted(entities.items()))})


def covers_unpublished(dcids: Collection[str], *, datasets: Sequence[str]) -> bool:
    """Whether the whole population `dcids` names reads, on a shape answered from
    `datasets`, as its side left unnamed: none of `datasets` publishes any of its parties on
    its side (`PartyRow.publishes`), with a whole marker or without one, those datasets hold some
    party on its side, and every one, totals aside, is of a class its markers elsewhere say
    it covers. The CRS publishes no total of its recipients and holds `official` and
    `residual` ones, so "all recipients" (`DACAllRecipients`, `official` and `residual`)
    reads as every recipient it holds. DAC2b publishes `DACAllRecipients` with no marker, so
    there it binds as a named party and reads as no whole population. The CRS holds private
    donors beside official ones, so "official donors" (`DACOfficialDonors`, `official` alone)
    never reads as every donor it holds; DAC1 holds official donors alone, so there it does."""
    index = get_index()
    rows = [index.parties.get(dcid) for dcid in dcids]
    if not datasets or not rows or any(row is None for row in rows):
        return False
    parties = cast(list[PartyRow], rows)
    axis = parties[0].sole_axis()
    if any(row.publishes(axis, datasets) for row in parties):
        return False
    covers = {cls for row in parties for marker in row.whole for cls in marker.covers}
    classes = side_classes(axis, datasets=datasets)
    return bool(classes) and set(classes) <= covers


def side_classes(axis: RoleAxis, *, datasets: Sequence[str]) -> tuple[PartyClass, ...]:
    """The classes of party, totals aside, that `datasets` hold on `axis`, sorted
    (`CodeIndex.side_classes`)."""
    by_side = get_index().side_classes
    return tuple(sorted({cls for dataset in datasets for cls in by_side.get((axis, dataset), ())}))


def whole_total_unpublished_facts(
    spec: Spec, whole_named: Mapping[str, tuple[str, ...]]
) -> list[DepartureFact]:
    """`whole_total_unpublished` for each whole population a text of the part names
    (`whole_named`, each text with the dcids it names) whose side `spec`'s identity leaves
    unnamed because the dataset elected for it publishes no total of the population and
    holds no party outside it there (`covers_unpublished`). `asked` is the text, `shown`
    the dataset's name, `over` the side and `whole_of` the classes of party the dataset holds
    there (`side_classes`). A door that holds no words of the request (a resubmit) names no
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
        axis = parties[dcids[0]].sole_axis()
        if axis in bound or not covers_unpublished(dcids, datasets=(dataset,)):
            continue
        facts.append(
            DepartureFact(
                code="whole_total_unpublished",
                binding_ref=role_ref(axis),
                asked=text,
                shown=shown,
                over=axis,
                whole_of=side_classes(axis, datasets=(dataset,)),
            )
        )
    return facts


def named_whole(text: str, resolved: str | None) -> tuple[str, ...] | None:
    """The parties a text of the part names as a whole population, `None` when it names
    none: the members of the whole total set whose phrase it is, else the one party it
    resolved to (`resolved`) when some dataset marks that party as covering the `official`
    class. "All donors" is a phrase of the official and private donors' set; "official
    donors" resolves to `DACOfficialDonors`; "private donors" resolves to
    `DACPrivateDonors`, which covers private donors alone, and names a party, not a
    population."""
    index = get_index()
    if (whole_set := index.whole_set(text)) is not None:
        return whole_set.members
    party = None if resolved is None else index.parties.get(resolved)
    if party is None or not any(POPULATION_CLASS in marker.covers for marker in party.whole):
        return None
    return (party.dcid,)


def named_whole_label(dcids: Collection[str]) -> str | None:
    """The label a side that reads `dcids` whole carries (`WholeReading.label`): the phrase
    of the whole total set they are the members of, else the one party's label. `None` when
    they are neither one party nor one set's members."""
    index = get_index()
    members = frozenset(dcids)
    if len(members) > 1:
        whole_set = whole_total_set_of(members)
        return None if whole_set is None else whole_set.phrase
    party = index.parties.get(next(iter(members))) if members else None
    return None if party is None else party.label or party.dcid


def population_totals(axes: Collection[RoleAxis]) -> tuple[str, ...]:
    """The parties some dataset marks as covering the `official` class on one of `axes`, by
    dcid: the totals a side the part totals can bind (`place_roles.totalled_axes`)."""
    by_axis = get_index().population_totals_by_axis
    return tuple(sorted(dcid for axis in set(axes) for dcid in by_axis.get(axis, ())))


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
    side = side_classes(reading.axis, datasets=(dataset,))
    private = private_outside(reading.axis, reading.whole_of, side)
    differs = " includes private donors and" if private else ""
    return (
        f"The source publishes the total of {population} as {reading.label}{holds}; "
        f"a sum of the listed {noun}s{differs} differs from it."
    )
