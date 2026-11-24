"""The donor and recipient axes: a flow side as the part names it, its total, or its members.

A part that names parties on a flow side offers `<axis>:named`, the parties as posted, which
the part holds, each total that can stand for the side's whole as `<axis>:<dcid>`, and
`<axis>:members`, the side summed over its members (`summed_members`), stated as an open
role. The side's whole (`AxisTable.whole`) is the first of these the part's table
publishes:

- on a composite whose component reads the bilateral destinations of its table
  (`ComponentRecipe.destinations`), the recipient side's population total
  (`BilateralSource.rebound`, "Developing Countries, Total"), the side the composite reads;
- the containing total its dataset serves on the side (`CONTAINING_TOTALS`, DAC1's "DAC
  countries, total");
- the one total the code index marks whole on the side for the part's years
  (`whole_reading`, DAC2a's "All Recipients, Total");
- else the members' sum.

Where the code index marks two totals whole, the side declares no whole
(`whole_ambiguous`). A containing total holds only the members of its group over the part's
years, so it is unavailable (`part_outside_whole`) where the part names a party outside
that group: DAC1's DAC countries' total holds no non-DAC donor and no multilateral. The
totals the code index marks whole are then offered beside it. A published total is
otherwise available where the coverage snapshot the table is read from holds a series of
the part's variables at it beside the other side's places (`series_held`); unavailable
(`option_not_covered`) where it holds none or no variable is elected there, and
(`coverage_unknown`) where the snapshot cannot say. A side the part leaves open or names
only by its whole already is whole: its one option is the one it holds, so a whole request
refuses `part_is_whole`. A side the dataset publishes no party on is absent. The recipient
side also offers its dataset's total of both sides (`BilateralSource.spanning`, DAC2a's "All
Recipients, Total"), outside the whole's alternatives, for a provider whole that rebinds the
side (`AxisTable.whole_rebind`).

A donor side summed over its members counts every reporting donor, the multilateral
agencies among them, so it conflicts (`agencies_counted_twice`) with the provider options
that count a donor's money through those agencies, the inclusive scope and total ODA.

These rules are not in `rules.AXIS_RULES`: a fresh answer does not show them, and only a
whole request reads them (`whole.derive_whole`).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import date

from code_index.composites import (
    DECLARED_COMPOSITES,
    CompositeNotServable,
    composite_ref,
    destination_population,
)
from code_index.declarations import CONTAINING_TOTALS
from code_index.query import CodeIndex
from code_index.scope_conventions import BILATERAL_SOURCES
from code_index.shared import get_index
from dfo_platform.roles import role_key_axis
from search.catalogue.counts import COUNTRY_PREFIX
from search.engine.region.coverage_series import series_held, series_span
from search.engine.retrieval.bound_values import identity_dataset, served_members
from search.engine.retrieval.whole_reading import identity_years, whole_reading
from search.models import (
    BaseIdentity,
    IdentityWindow,
    Option,
    OptionAvailability,
    RoleAxis,
    ScopeAxis,
    SideBinding,
    TimeWindow,
)
from search.models.vocab import SCOPE_OPTION_ID_PATTERN
from search.scope.provider import _window_days
from search.scope.region import policy_members_between, policy_members_throughout
from search.scope.state import AxisRecord
from search.scope.tables import AxisTable, Conflict, ScopeFacts

_SIDE_AXIS: dict[ScopeAxis, RoleAxis] = {"donor": "origin", "recipient": "destination"}
_REBOUND = {row.dataset: row.rebound for row in BILATERAL_SOURCES}
_SPANNING = {row.dataset: row.spanning for row in BILATERAL_SOURCES}
_OPTION_ID = re.compile(SCOPE_OPTION_ID_PATTERN)
_PER_DONOR_SCOPES = ("provider:inclusive", "provider:contributions")
"""The provider options that count a donor's money beyond its own bilateral flows, which a
sum over every reporting donor counts twice: the multilateral agencies report as donors too."""
AGENCIES_COUNTED_TWICE = "agencies_counted_twice"
"""Why a donor side summed over its members conflicts with `_PER_DONOR_SCOPES`."""


def side_rule(axis: ScopeAxis):
    """The rule of the side axis `axis` (`donor` or `recipient`)."""

    def rule(base: BaseIdentity, record: AxisRecord | None, facts: ScopeFacts) -> AxisTable | None:
        return _table(axis, base, facts)

    return rule


def _table(axis: ScopeAxis, base: BaseIdentity, facts: ScopeFacts) -> AxisTable | None:
    flow = _SIDE_AXIS[axis]
    keys = [key for key in base.entities if role_key_axis(key) == flow]
    if len(keys) > 1:
        return None
    table_identity, dataset, rebound = _table_of(base, flow)
    if dataset is None:
        return None
    if not keys or flow in (base.open_roles or ()):
        # A side left open, or totalled where the dataset publishes parties on it, is whole.
        published = (flow, dataset) in get_index().side_classes
        spanning = _spanning(axis, flow, dataset, [], table_identity, facts)
        return _whole_already(axis, flow, [], spanning) if published else None
    (key,) = keys
    named = sorted(base.entities[key])
    spanning = _spanning(axis, key, dataset, named, table_identity, facts)
    containing = _containing(dataset, flow)
    marked = () if rebound or containing else _marked(dataset, flow, base.requested_window)
    total = rebound or containing or (marked[0] if len(marked) == 1 else None)
    held = f"{axis}:named"
    members = f"{axis}:members"
    if named == [total] or (len(marked) > 1 and len(named) == 1 and named[0] in marked):
        return _whole_already(axis, key, named, spanning)
    outside = (
        _outside(named, containing, base.requested_window, _years(table_identity, facts))
        if rebound is None and containing is not None
        else ()
    )
    totals = [*marked] if len(marked) > 1 else [total] if total is not None else []
    if outside:
        totals += [dcid for dcid in _marked(dataset, flow, base.requested_window) if dcid != total]
    options = [
        Option(
            option_id=held,
            label="as named",
            binding=SideBinding(role=key, refs=named),
            availability=OptionAvailability(status="available"),
        ),
        *(
            Option(
                option_id=f"{axis}:{dcid}",
                label=_label(dcid),
                binding=SideBinding(role=key, refs=[dcid]),
                availability=OptionAvailability(
                    status="unavailable",
                    code="part_outside_whole",
                    detail={"total": dcid, "outside": ",".join(outside)},
                )
                if outside and dcid == containing
                else _availability(table_identity, key, dcid, facts),
            )
            for dcid in totals
            if _OPTION_ID.fullmatch(f"{axis}:{dcid}")
        ),
        Option(
            option_id=members,
            label="every member",
            binding=SideBinding(role=key, refs=[]),
            availability=OptionAvailability(status="available"),
        ),
    ]
    if spanning is not None and spanning.option_id not in {o.option_id for o in options}:
        options.append(spanning)
    whole = None if len(marked) > 1 else members if total is None else f"{axis}:{total}"
    if whole is not None and not any(option.option_id == whole for option in options):
        whole = None
    return AxisTable(
        axis=axis,
        options=tuple(options),
        preference=(held,),
        held=held,
        rules={
            option.option_id: "side_as_named" if option.option_id == held else "side_whole"
            for option in options
        },
        whole=whole,
        whole_alternatives=tuple(
            option.option_id
            for option in options
            if option.option_id not in (held, whole) and option is not spanning
        ),
        whole_cause="whole_ambiguous",
        conflicts=tuple(
            Conflict(members, scope, AGENCIES_COUNTED_TWICE) for scope in _PER_DONOR_SCOPES
        )
        if axis == "donor"
        else (),
    )


def _whole_already(
    axis: ScopeAxis, key: str, named: list[str], spanning: Option | None
) -> AxisTable:
    """The table of a side the part names by its whole, or leaves open: the option it holds,
    which is the whole, so a whole request on the side refuses `part_is_whole`, and the
    dataset's total of both sides (`_spanning`) where it is another, which a provider whole
    rebinds the side to (`AxisTable.whole_rebind`)."""
    whole = f"{axis}:{named[0]}" if named else f"{axis}:members"
    if not _OPTION_ID.fullmatch(whole):
        whole = f"{axis}:named"
    held = Option(
        option_id=whole,
        label=_label(named[0]) if named else "every member",
        binding=SideBinding(role=key, refs=named),
        availability=OptionAvailability(status="available"),
    )
    extra = (spanning,) if spanning is not None and spanning.option_id != whole else ()
    return AxisTable(
        axis=axis,
        options=(held, *extra),
        preference=(whole,),
        held=whole,
        rules={whole: "side_whole", **{o.option_id: "side_whole" for o in extra}},
        whole=whole,
    )


def _spanning(
    axis: ScopeAxis,
    key: str,
    dataset: str,
    named: list[str],
    table_identity: BaseIdentity,
    facts: ScopeFacts,
) -> Option | None:
    """The option binding the recipient side to its dataset's total of both sides
    (`BilateralSource.spanning`, DAC2a's "All Recipients, Total"), which a provider whole
    rebinds the side to (`ProviderWhole.recipient`); None on the donor side, on a dataset
    declaring none, or where the part names that total alone."""
    dcid = _SPANNING.get(dataset) if axis == "recipient" else None
    if dcid is None or named == [dcid] or not _OPTION_ID.fullmatch(f"{axis}:{dcid}"):
        return None
    return Option(
        option_id=f"{axis}:{dcid}",
        label=_label(dcid),
        binding=SideBinding(role=key, refs=[dcid]),
        availability=_availability(table_identity, key, dcid, facts),
    )


def _label(dcid: str) -> str:
    party = get_index().parties.get(dcid)
    return party.label if party is not None and party.label else dcid


def _table_of(base: BaseIdentity, flow: RoleAxis) -> tuple[BaseIdentity, str | None, str | None]:
    """The identity coverage is read for, its dataset, and the population total a composite
    reads on `flow` (`BilateralSource.rebound`), None when it reads none."""
    document = base.model_dump(mode="json", exclude_none=True)
    try:
        ref = composite_ref(document)
    except CompositeNotServable:
        return base, None, None
    if ref is None:
        return base, identity_dataset(base), None
    composite = DECLARED_COMPOSITES[ref]
    bilateral = [r for r in composite.components if r.destinations == "bilateral"]
    dataset = bilateral[0].dataset if bilateral else composite.components[0].dataset
    rebound = _REBOUND.get(dataset) if bilateral and flow == "destination" else None
    facets = {k: v for k, v in (base.facets or {}).items() if k != "provenance"}
    table_identity = base.model_copy(update={"facets": facets or None})
    return table_identity, dataset, rebound


def summed_members(identity: BaseIdentity, role: str) -> list[str] | None:
    """The parties a whole that leaves `identity`'s side `role` open sums: every party and
    residual its dataset publishes on that side, its totals left out, as an open read
    leaves them out, or on a composite reading the bilateral destinations the population it
    reads (`RolePopulation.members`). None where the side is not a flow side or no dataset is elected."""
    flow = role_key_axis(role)
    if flow is None:
        return None
    _, dataset, rebound = _table_of(identity, flow)
    if dataset is None:
        return None
    if rebound is not None:
        return list(destination_population(get_index(), dataset).members)
    return sorted(
        party.dcid
        for party in get_index().parties.values()
        if (dataset, flow) in party.published and party.kind_in(dataset) != "total"
    )


def _containing(dataset: str, flow: RoleAxis) -> str | None:
    held = CONTAINING_TOTALS.get((dataset, flow))
    return None if held is None else held.total


def _outside(
    named: Sequence[str], total: str, window: IdentityWindow, years: tuple[int, int] | None
) -> tuple[str, ...]:
    """The parties of `named` a containing total does not hold over `window`: those whose
    members (a country itself, or the members of the group it totals on any day of the
    window) are not all members of the group `total` totals on every day of it
    (`region.policy_members_throughout`), as Korea is not of the DAC's before 2010. A window
    open at either end reads the part's own data `years` there, where the snapshot holds
    them."""
    first, last = _window_days(window)
    if years is not None and first == date.min:
        first = date(years[0], 1, 1)
    if years is not None and last == date.max:
        last = date(years[1], 12, 31)
    index = get_index()
    held = _group_members(total, index, lambda group: policy_members_throughout(group, first, last))
    return tuple(
        party
        for party in named
        if not (
            members := _group_members(
                party, index, lambda group: policy_members_between(group, first, last)
            )
        )
        or not members <= held
    )


def _years(identity: BaseIdentity, facts: ScopeFacts) -> tuple[int, int] | None:
    """The first and last year the snapshot holds a series of `identity`'s variables at the
    places it names (`series_span`), None where it holds none."""
    bound = {
        axis: dcids
        for role, dcids in identity.entities.items()
        if (axis := role_key_axis(role)) is not None
    }
    variables = served_members(identity)
    if not variables:
        return None
    return series_span(facts.coverage.index, variables=variables, bound=bound)


def _group_members(
    party: str, index: CodeIndex, members_of: Callable[[str], frozenset[str]]
) -> frozenset[str]:
    """What a total of `party` sums: the country itself, or `members_of` the group it totals
    (`PartyRow.totals`); empty for any other party."""
    if party.startswith(COUNTRY_PREFIX):
        return frozenset((party,))
    row = index.parties.get(party)
    if row is None or row.totals is None:
        return frozenset()
    return members_of(row.totals)


def _marked(dataset: str, flow: RoleAxis, window: IdentityWindow) -> tuple[str, ...]:
    """The totals the code index marks whole on `flow` in `dataset` over `window`."""
    from search.engine.region.whole_totals import population_totals

    return tuple(
        dcid
        for dcid in population_totals((flow,))
        if whole_reading((dcid,), datasets=(dataset,), years=identity_years(window)) is not None
    )


def _availability(
    identity: BaseIdentity, key: str, total: str | None, facts: ScopeFacts
) -> OptionAvailability:
    """Whether the side's `total` serves the part: available where the snapshot holds a
    series of `identity`'s variables at it on the side `key` beside the other side's places,
    or where no total is bound (the side summed over its members). Unavailable with
    `option_not_covered` where the snapshot holds no such series or the code index elects
    no variable there, and with `coverage_unknown` where the snapshot cannot say, since a
    whole stated on an unread total would read as covered."""
    if total is None:
        return OptionAvailability(status="available")
    whole = identity.model_copy(update={"entities": {**identity.entities, key: [total]}})
    variables = served_members(whole)
    if not variables:
        return OptionAvailability(
            status="unavailable", code="option_not_covered", detail={"total": total}
        )
    bound = {
        axis: dcids
        for role, dcids in whole.entities.items()
        if (axis := role_key_axis(role)) is not None
    }
    window = whole.requested_window
    span = (
        TimeWindow(start_year=window.start_year, end_year=window.end_year)
        if window.kind == "range"
        else None
    )
    held = series_held(facts.coverage.index, variables=variables, bound=bound, window=span)
    if held is None:
        return OptionAvailability(
            status="unavailable", code="coverage_unknown", detail={"total": total}
        )
    if not held:
        return OptionAvailability(
            status="unavailable", code="option_not_covered", detail={"total": total}
        )
    return OptionAvailability(status="available")
