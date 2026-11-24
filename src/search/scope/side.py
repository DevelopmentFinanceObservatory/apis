"""The donor and recipient axes: a flow side as the part names it, or its whole.

A part that names parties on a flow side offers two options there: `<axis>:named`, the
parties as posted, which the part holds, and `<axis>:whole`, the side's whole. The whole is
the first of these the part's table publishes:

- on a composite whose component reads the bilateral destinations of its table
  (`ComponentRecipe.destinations`), the recipient side's population total
  (`BilateralSource.rebound`, "Developing Countries, Total"), the side the composite reads;
- the containing total its dataset serves on the side (`CONTAINING_TOTALS`, DAC1's "DAC
  countries, total");
- the one total the code index marks whole on the side for the part's years
  (`whole_reading`, DAC2a's "All Recipients, Total");
- else the side left open, summed over its members.

A containing total holds only the members of its group, so it is unavailable
(`part_outside_whole`) where the part names a party outside that group: DAC1's DAC countries'
total holds no non-DAC donor and no multilateral. A published total is otherwise available where the coverage snapshot the table is read from holds a
series of the part's variables at it beside the other side's places (`series_held`), so its
availability is stated in the same build as every other option's. A side the part leaves open
or names only by its whole already is whole, and the axis is absent.

These rules are not in `rules.AXIS_RULES`: a fresh answer does not show them, and a part
reads them only where an entry posts their axis (`build.SIDE_RULES`).
"""

from __future__ import annotations

from collections.abc import Sequence

from code_index.composites import DECLARED_COMPOSITES, CompositeNotServable, composite_ref
from code_index.declarations import CONTAINING_TOTALS
from code_index.scope_conventions import BILATERAL_SOURCES
from code_index.shared import get_index
from dfo_platform.roles import role_key_axis
from search.engine.region.coverage_series import series_held
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
from search.scope.state import AxisRecord
from search.scope.tables import AxisTable, ScopeFacts

_SIDE_AXIS: dict[ScopeAxis, RoleAxis] = {"donor": "origin", "recipient": "destination"}
_REBOUND = {row.dataset: row.rebound for row in BILATERAL_SOURCES}


def side_rule(axis: ScopeAxis):
    """The rule of the side axis `axis` (`donor` or `recipient`)."""

    def rule(base: BaseIdentity, record: AxisRecord | None, facts: ScopeFacts) -> AxisTable | None:
        return _table(axis, base, facts)

    return rule


def _table(axis: ScopeAxis, base: BaseIdentity, facts: ScopeFacts) -> AxisTable | None:
    flow = _SIDE_AXIS[axis]
    keys = [key for key in base.entities if role_key_axis(key) == flow]
    if len(keys) != 1 or flow in (base.open_roles or ()):
        return None
    (key,) = keys
    named = sorted(base.entities[key])
    table_identity, dataset, rebound = _table_of(base, flow)
    if dataset is None:
        return None
    containing = _containing(dataset, flow)
    total = rebound or containing or _marked(dataset, flow, base.requested_window)
    if total is not None and named == [total]:
        return None
    whole_refs = [] if total is None else [total]
    held = f"{axis}:named"
    whole = f"{axis}:whole"
    outside = _outside(named, containing) if rebound is None and containing is not None else ()
    availability = (
        OptionAvailability(
            status="unavailable",
            code="part_outside_whole",
            detail={"total": containing or "", "outside": ",".join(outside)},
        )
        if outside
        else _availability(table_identity, key, whole_refs, facts)
    )
    options = (
        Option(
            option_id=held,
            label="as named",
            binding=SideBinding(role=key, refs=named),
            availability=OptionAvailability(status="available"),
        ),
        Option(
            option_id=whole,
            label="every member" if total is None else total,
            binding=SideBinding(role=key, refs=whole_refs),
            availability=availability,
        ),
    )
    return AxisTable(
        axis=axis,
        options=options,
        preference=(held,),
        held=held,
        rules={held: "side_as_named", whole: "side_whole"},
        whole=whole,
    )


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


def _containing(dataset: str, flow: RoleAxis) -> str | None:
    held = CONTAINING_TOTALS.get((dataset, flow))
    return None if held is None else held.total


def _outside(named: Sequence[str], total: str) -> tuple[str, ...]:
    """The parties of `named` a containing total does not hold: those whose members
    (`provider._members`, a country itself or the members of the group it totals) are not
    all members of the group `total` totals in the membership policy."""
    from search.scope.provider import _members

    index = get_index()
    held = _members(total, index)
    return tuple(
        party for party in named if not (members := _members(party, index)) or not members <= held
    )


def _marked(dataset: str, flow: RoleAxis, window: IdentityWindow) -> str | None:
    """The one total the code index marks whole on `flow` in `dataset` over `window`."""
    from search.engine.region.whole_totals import population_totals

    readings = [
        dcid
        for dcid in population_totals((flow,))
        if whole_reading((dcid,), datasets=(dataset,), years=identity_years(window)) is not None
    ]
    return readings[0] if len(readings) == 1 else None


def _availability(
    identity: BaseIdentity, key: str, refs: Sequence[str], facts: ScopeFacts
) -> OptionAvailability:
    """Available where the snapshot holds a series of `identity`'s variables at `refs` on the
    side `key` beside the other side's places, or where no total is bound."""
    if not refs:
        return OptionAvailability(status="available")
    whole = identity.model_copy(update={"entities": {**identity.entities, key: list(refs)}})
    variables = served_members(whole)
    if not variables:
        return OptionAvailability(status="available")
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
    if held is False:
        return OptionAvailability(
            status="unavailable", code="option_not_covered", detail={"total": refs[0]}
        )
    return OptionAvailability(status="available")
