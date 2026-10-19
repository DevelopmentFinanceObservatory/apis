"""Whether a role-bearing shape holds a series at given places, read from the request's
coverage snapshot.

The availability read asks it of the places a part names (`availability.series_at_places`),
and binding repair asks it of a whole total beside them (`place_roles.holds_whole_total`),
under the slot bindings settled for the total's role keys, so a side binds the source's total only where a series of
the variables the bindings elect reaches it. A Spec that lists a side member by member asks
the same of its own variables before it states the side's total
(`whole_totals.whole_total_warnings`, `series_held`).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

from code_index.shared import get_index
from search.engine.extract import DateRequest
from search.engine.region.place_set import PlaceSet
from search.engine.retrieval.bound_values import (
    _bound_value_filter_facts,
    bound_values_from_bindings,
    part_member_rows,
)
from search.engine.retrieval.member_facts import ArcsMemberFacts, IndexMemberFacts, MemberFacts
from search.engine.retrieval.role_scope import _COVERAGE_ROLE_FOR_AXIS
from search.engine.shape import ShapeDraft
from search.models import EntityRoleKey, RoleAxis, TimeWindow

if TYPE_CHECKING:
    from search.engine.bind import SlotBindingDraft
    from search.engine.region.state import ResolveState
    from series_coverage.index import CoverageIndex


def requested_window(date_request: DateRequest | None) -> TimeWindow | None:
    """The years a part asks for, `None` when it names none or asks for the latest."""
    return None if date_request is None or date_request.latest else date_request.window


def series_held(
    coverage: CoverageIndex,
    *,
    variables: Iterable[str],
    bound: Mapping[RoleAxis, Collection[str]],
    window: TimeWindow | None,
) -> bool | None:
    """Whether `coverage` holds a series of one of `variables` at the places `bound` gives
    each flow axis, within `window`. `None` when no flow axis binds a place or the coverage
    index holds none of `variables`."""
    by_role = {
        _COVERAGE_ROLE_FOR_AXIS[axis]: frozenset(dcids)
        for axis, dcids in bound.items()
        if axis in _COVERAGE_ROLE_FOR_AXIS and dcids
    }
    covered = [dcid for dcid in variables if coverage.covers(dcid)]
    if not by_role or not covered:
        return None
    return bool(
        coverage.filter_candidates(
            variables=covered,
            bound=by_role,
            first_year=None if window is None else window.start_year,
            last_year=None if window is None else window.end_year,
        )
    )


def holds_series(
    state: ResolveState,
    shape: ShapeDraft,
    sets: Mapping[EntityRoleKey, Sequence[PlaceSet]],
    *,
    bindings: Sequence[SlotBindingDraft] = (),
) -> bool | None:
    """Whether role-bearing `shape`'s family holds a series at `sets`, the place sets bound
    on each role, within the part's requested years (`series_held`). A set gives its node
    and its members. The family is the rows its part may answer from (`part_member_rows`),
    else its recalled members, less those asserting a value `bindings` does not bind
    (`bound_values._bound_value_filter_facts`). `None` when no flow role binds a place or
    the coverage index holds no such member."""
    rows = part_member_rows(shape, get_index())
    members: dict[str, MemberFacts] = (
        {row.dcid: IndexMemberFacts(row=row) for row in rows}
        if rows
        else {dcid: ArcsMemberFacts(arcs=arcs) for dcid, arcs in (shape.sv_arc_facts or {}).items()}
    )
    elected = _bound_value_filter_facts(members, bound_values_from_bindings(bindings))
    return series_held(
        state.coverage.index,
        variables=elected,
        bound={
            axis: [
                dcid
                for place_set in placed
                for dcid in (*place_set.dcids("default"), *place_set.dcids("members"))
            ]
            for axis, placed in sets.items()
            if axis in _COVERAGE_ROLE_FOR_AXIS
        },
        window=requested_window(state.date_request),
    )
