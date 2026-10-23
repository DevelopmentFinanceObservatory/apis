"""``materialisation``: each definite part's stated coverage holds in the graph.

Reads, on each definite part's spec (``parts[].spec``):

- ``resolved.coverage.{kind, has_data, dimensions[].{label, count}, observation_count,
  window.{start_year, end_year}}`` and ``resolved.coverage_stamp``;
- ``resolved.facets.<axis>.{ref, request_basis}`` for the provenance, measurementMethod and
  unit axes;
- ``resolved.entities[].{ref, role}``, ``resolved.observations.series[].entities[].ref``
  and ``resolved.scope.{axis, members[].ref}``;
- ``definition`` and ``definition.identity.requested_window.kind``;
- ``spec_id`` with ``diagnostics.warnings[].{code, spec_id}`` and
  ``parts[].warnings[].{code, spec_id}``.

A bare answer passes on ``has_data`` and a breadth answer on every dimension counting
something. An exact count is checked, within 5 percent, against the oracle's own count of
the facets the answer states it serves, inside the coverage window, and under a ``latest``
request at each StatVar's one period. A role-bearing answer's StatVar serves the year the
coverage index elects for it over those facets and the bound roles
(``CoverageIndex.elected_years``), from the series holding that year. A place-keyed
answer's StatVar, and a role-bearing one the index elects no year for, serves its latest
period from the series reaching it. A series whose
stated facets end before the coverage window does also counts the points after its last
one from the same source's estimates continuing it (``materialise.facets.continuation``),
which the answer serves without stating their method. The StatVars are those
the responding app states for the spec (``base.served_stat_vars``). StatVars and places
listed twice count once. A non-role answer is counted from ``ctx.graph`` at each StatVar and
served place (``_served_places``). A role-bearing answer is counted from
``ctx.coverage.snapshot()``, one series per StatVar and bound role tuple, as Search reads it.

A part the check cannot count is n/a, with the reason in the detail: a role-bearing answer
confirmed against another coverage stamp, or mixing ``observationAbout`` with a flow role; a
converted answer whose conversion left rows out of its count; a non-role answer with a
partial preview whose planned places are not on the wire; a non-role answer naming no place
at all; and a non-role answer at whose places the graph holds no records, as at a region
node. A failed graph, coverage or members read (``base.READ_ERRORS``), or a missing client,
is an infrastructure result. Any other error propagates.

The check reads the graph once per definite part, so it registers ``modal_only``: the driver
runs it on the modal sample alone.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from dc_graph import Facet
from materialise.facets import Cadence, at_cadence, cadences_by_member, continuation
from materialise.window import RequestedWindow, latest_period
from search.engine.region.codes import (
    CURRENCY_UNMAPPED_OMITTED,
    OBSERVATIONS_PROBE_FAILED,
    OBSERVATIONS_SERIES_TRUNCATED,
    OBSERVATIONS_TIMED_OUT,
)
from search.engine.retrieval.role_scope import _COVERAGE_ROLE_FOR_AXIS, facet_from_coverage
from search.eval.cases import Case, StatusExpect
from search.eval.checks.base import (
    PASS,
    READ_ERRORS,
    MembersRefused,
    PartView,
    Result,
    ScoreContext,
    Verdict,
    by_part,
    expected_parts,
    fail,
    infra_error,
    part_pairs,
    read_path,
    register,
    run_part,
    served_stat_vars,
    unreached,
)
from search.eval.checks.specs import SpecView
from search.models import TimeWindow, in_window
from series_coverage.coverage_types import BoundRoles
from series_coverage.coverage_types import Role as CoverageRole
from series_coverage.shared import CoverageSnapshot

_FACET_AXIS_FIELDS = {
    "provenance": "provenance_id",
    "measurementMethod": "measurement_method",
    "unit": "unit",
}
"""The `Facet` field holding each axis of `resolved.facets`."""

type _Stated = tuple[str | None, bool]
"""One axis of `resolved.facets`: its `ref`, and whether a `request_basis` binds it."""

type _FacetGroup = tuple[tuple[str | None, ...], Cadence | None]
"""A facet's provenance, measurement method and unit, and its cadence
(`materialise.facets.cadences_by_member`): what the facets one dispatch serves share."""

_PARTIAL_PREVIEW_CODES = frozenset(
    {OBSERVATIONS_PROBE_FAILED, OBSERVATIONS_TIMED_OUT, OBSERVATIONS_SERIES_TRUNCATED}
)
"""Warnings a preview carries when it delivered some of its planned series, whose count the
engine states over every planned series."""


def _axis_admits(stated: _Stated | None, value: str | None) -> bool:
    """Whether a facet carrying `value` on an axis is one the answer states it serves there.

    An axis stating a `ref` admits the facets carrying it, and an axis the answer omits
    admits the facets declaring no value on it. An axis stating `per_series` names no
    value, and a unit bound through the request form states the currency it converts to,
    so both admit every value.
    """
    if stated is None:
        return value is None
    ref, converts = stated
    if ref is None or converts:
        return True
    return value == ref


def _stated_groups(
    facets: Sequence[Facet], stated: Mapping[str, _Stated]
) -> dict[_FacetGroup, list[Facet]]:
    """The facets of one series that `stated` admits on every axis (`_axis_admits`), by
    group. A group is one cadence of one provenance, measurement method and unit, since a
    dispatch serves its series at one observation period (`materialise.facets.served_facets`).
    A series whose admitted facets form several groups is one the answer does not pin to a
    single group."""
    (cadence_of,) = cadences_by_member([facets])
    groups: dict[_FacetGroup, list[Facet]] = {}
    for facet in facets:
        values = tuple(getattr(facet, field) for field in _FACET_AXIS_FIELDS.values())
        if all(
            _axis_admits(stated.get(axis), value)
            for axis, value in zip(_FACET_AXIS_FIELDS, values, strict=True)
        ):
            key = (values, cadence_of.get(facet.facet_id or ""))
            groups.setdefault(key, []).append(facet)
    return groups


def _group_points(group: Sequence[Facet], *, window: TimeWindow | None) -> int:
    """The points one series serves from `group` inside `window`, one per distinct date
    across the group's facets. Without a window, a group holding a facet whose dates are
    incomplete counts each facet's observations."""
    if window is None and any(
        f.obs_count > 0 and not (f.dates and f.dates_complete) for f in group
    ):
        return sum(f.obs_count for f in group)
    return len({d for f in group for d in f.dates if in_window(d, window)})


def _series_points(
    group: Sequence[Facet], facets: Sequence[Facet], *, window: TimeWindow | None
) -> int:
    """The points one series serves from `group` (`_group_points`) and from the facets of
    the series' `facets` continuing it past its last period inside `window`, chosen as the
    engine and a mint choose them (`materialise.facets.continuation`, at the cadence of the
    group's lead facet). A continuing facet serves only its dates after the group's last
    period (`materialise.columnar.continuation_mask`)."""
    points = _group_points(group, window=window)
    if window is None or window.end_year is None:
        return points
    (cadence_of,) = cadences_by_member([facets])
    cadence = cadence_of.get(group[0].facet_id or "")
    continuing = continuation(
        group,
        facets,
        window=RequestedWindow(
            kind="range", start_year=window.start_year, end_year=window.end_year
        ),
        serves=lambda facet_id: at_cadence(facet_id, cadence=cadence, cadence_of=cadence_of),
    )
    last = _group_latest(group)
    later = {
        d
        for f in continuing
        for d in f.dates
        if in_window(d, window) and (last is None or d > last)
    }
    return points + len(later)


def _group_latest(group: Sequence[Facet]) -> str | None:
    """The newest period `group`'s facets state (`materialise.window.latest_period`), the
    one a read of its series alone serves under `latest`."""
    return latest_period(f.latest_date or max(f.dates, default=None) for f in group)


def _add_totals(totals: int, other: int) -> int:
    """The bitset of every sum of a total in `totals` and one in `other`."""
    sums = 0
    while other:
        low = other & -other
        sums |= totals << (low.bit_length() - 1)
        other ^= low
    return sums


def _latest_totals(series_latest: Sequence[Sequence[str | None]], window: TimeWindow | None) -> int:
    """The totals one StatVar's series give under `latest`, as a bitset, from each series'
    stated groups' latest periods (`_group_latest`).

    The StatVar serves one period, the newest its series' groups reach, and one point from
    each series whose group reaches it (`materialise.window.latest_period`, the rule
    `search.engine.retrieval.read_plan.PairPlan.served` applies). A series whose group ends
    earlier serves nothing, and a period outside `window` gives nothing. Each period `p` a
    group reaches is a branch: every series takes a group ending no later than `p`, and at
    least one takes a group ending at it.
    """
    totals = 0
    for period in {p for groups in series_latest for p in groups if p is not None}:
        branch = 1
        for groups in series_latest:
            points = 0
            for g in groups:
                if g is None or latest_period((g, period)) == period:
                    points |= 1 << (g == period)
            branch = _add_totals(branch, points)
        branch &= ~1
        if branch:
            totals |= branch if in_window(period, window) else 1
    if all(None in groups for groups in series_latest):
        totals |= 1
    return totals


def _elected_totals(
    series_groups: Sequence[Sequence[Sequence[Facet]]], year: int, window: TimeWindow | None
) -> int:
    """The totals one StatVar's series give under `latest` when the StatVar serves the
    elected `year`, as a bitset, from each series' stated groups.

    A series serves one point when its group holds a point in `year` and nothing
    otherwise (`search.engine.retrieval.read_plan.PairPlan.served`), so a series left with
    several groups branches on each. A year outside `window` gives nothing.
    """
    if not in_window(str(year), window):
        return 1
    at_year = TimeWindow(start_year=year, end_year=year)
    totals = 1
    for groups in series_groups:
        points = 0
        for group in groups:
            points |= 1 << min(1, _group_points(group, window=at_year))
        totals = _add_totals(totals, points or 1)
    return totals


def _reachable_totals(
    facets_by_series: Sequence[tuple[str, Sequence[Facet]]],
    stated: Mapping[str, _Stated],
    *,
    window: TimeWindow | None,
    latest: bool,
    elected: Mapping[str, int],
) -> int | None:
    """The totals the stated facets can give across `facets_by_series`, each series keyed by
    its StatVar, as a bitset whose bit `t` is set when one stated group per series sums to
    `t` observations, or None when the graph holds no facet for any series.

    Each series serves the points of one stated group and of the estimates continuing it
    (`_stated_groups`, `_series_points`), and nothing when no facet matches. A series
    pinned to one group adds its points to every total, and a series left with several
    groups branches into one total per group.
    Under `latest` each StatVar's series count together: at the year `elected` holds for
    it (`_elected_totals`), and for a StatVar absent from `elected` at its latest period
    (`_latest_totals`). The set holds at most one bit per observation the largest groups
    hold together.
    """
    if not any(facets for _sv, facets in facets_by_series):
        return None
    totals = 1
    if latest:
        by_sv: dict[str, list[list[list[Facet]]]] = {}
        for sv, facets in facets_by_series:
            by_sv.setdefault(sv, []).append(list(_stated_groups(facets, stated).values()))
        for sv, series_groups in by_sv.items():
            if sv in elected:
                served = _elected_totals(series_groups, elected[sv], window)
            else:
                series_latest = [
                    [_group_latest(g) for g in groups] or [None] for groups in series_groups
                ]
                served = _latest_totals(series_latest, window)
            totals = _add_totals(totals, served)
        return totals
    for _sv, facets in facets_by_series:
        points = 0
        for group in _stated_groups(facets, stated).values():
            points |= 1 << _series_points(group, facets, window=window)
        totals = _add_totals(totals, points or 1)
    return totals


def _nearest_total(totals: int, count: int) -> int:
    """The total in the non-empty bitset `totals` nearest to `count`, the smaller on a tie."""
    below = (totals & ((1 << (count + 1)) - 1)).bit_length() - 1
    above_bits = totals >> count
    above = count + (above_bits & -above_bits).bit_length() - 1 if above_bits else -1
    if below < 0:
        return above
    if above < 0 or count - below <= above - count:
        return below
    return above


def _count_result(totals: int, expected_count: int) -> Result:
    """Score `expected_count` against the reachable total in `totals` nearest to it, within
    5 percent."""
    observed = _nearest_total(totals, expected_count)
    reachable = totals.bit_count()
    nearest = f", nearest of {reachable} reachable totals" if reachable > 1 else ""
    tolerance = max(1, expected_count * 0.05)
    if abs(observed - expected_count) > tolerance:
        return fail(
            f"exact: count mismatch: observed={observed}, expected={expected_count}, "
            f"tolerance={tolerance:.1f}{nearest}"
        )
    return Result(
        Verdict.PASS, f"exact: observed={observed}, expected={expected_count} (within 5%){nearest}"
    )


def _unchecked(reason: str, detail: str) -> Result:
    """A definite part whose count the check cannot verify."""
    return Result(Verdict.NA, f"{reason}: {detail}")


def _role_series_facets(
    stat_vars: Sequence[str], bound: BoundRoles, coverage: CoverageSnapshot
) -> list[tuple[str, list[Facet]]]:
    """Each served role series' StatVar and facets from the coverage index, the series being
    one StatVar at one recipient and one donor surviving `bound`, as the answer's read plan
    serves them (`role_scope.pair_series`). A StatVar declaring no donor gives one series per
    recipient."""
    index = coverage.index
    per_pair = index.pair_facet_candidates(
        variables=stat_vars, bound=bound, pair_roles=("recipient", "donor")
    )
    metadata = index.facet_metadata
    return [
        (sv, [facet_from_coverage(facet, metadata.get(facet.facet_id)) for facet in facets])
        for (sv, _dcids), facets in per_pair.items()
    ]


def _stated_admits(
    stated: Mapping[str, _Stated], coverage: CoverageSnapshot
) -> Callable[[str, str], bool]:
    """Whether a coverage row's facet is one the answer states it serves on every axis
    (`_axis_admits`), given the row's variable and facet id, as `_stated_groups` admits
    the facets of a series."""
    metadata = coverage.index.facet_metadata

    def admits(_variable: str, facet_id: str) -> bool:
        meta = metadata.get(facet_id)
        return all(
            _axis_admits(stated.get(axis), getattr(meta, field, None))
            for axis, field in _FACET_AXIS_FIELDS.items()
        )

    return admits


def _spec_warning_codes(response: Mapping[str, Any], spec_id: str, ctx: ScoreContext) -> set[str]:
    """The warning codes concerning the spec `spec_id`: those naming it in the envelope or
    any part, and those naming no spec on a definite part carrying it. A part folded into an
    earlier part's spec shares that part's warnings, and the other parts of a conjunction
    keep their own."""
    at = "diagnostics.warnings[]"
    codes = {
        read_path(w, "code", ctx, at=at)
        for w in read_path(response, "diagnostics.warnings", ctx) or []
        if read_path(w, "spec_id", ctx, at=at) == spec_id
    }
    at = f"{PartView.path}.warnings[]"
    for part in read_path(response, "parts", ctx) or []:
        carries = (
            read_path(part, "status", ctx, at=PartView.path) == "definite"
            and read_path(part, "spec.spec_id", ctx, at=PartView.path) == spec_id
        )
        for w in read_path(part, "warnings", ctx, at=PartView.path) or []:
            named = read_path(w, "spec_id", ctx, at=at)
            if named == spec_id or (named is None and carries):
                codes.add(read_path(w, "code", ctx, at=at))
    return codes


def _stated_facets(spec: SpecView, ctx: ScoreContext) -> dict[str, _Stated]:
    stated = {}
    for axis in _FACET_AXIS_FIELDS:
        facet = spec.read(f"resolved.facets.{axis}", ctx)
        if facet is not None:
            at = f"{spec.at}.resolved.facets.{axis}"
            stated[axis] = (
                read_path(facet, "ref", ctx, at=at),
                read_path(facet, "request_basis", ctx, at=at) is not None,
            )
    return stated


def _refs(items: list[Mapping[str, Any]] | None, path: str, at: str, ctx: ScoreContext) -> list:
    return [read_path(item, path, ctx, at=at) for item in items or []]


def _served_places(
    spec: SpecView, ctx: ScoreContext, *, preview_complete: bool
) -> list[str] | None:
    """The places a non-role answer serves: those its observation series name when its
    preview is complete, or else its resolved entities and the members of a place scope it
    defaulted. A region answer names its member places only in its series, and an answer
    naming no place only in its scope. A partial preview's series leave out places the
    stated count includes, so when they name a place outside the entities and scope, as a
    region answer's do, the planned places are not on the wire and this returns None."""
    resolved = f"{spec.at}.resolved"
    series_at = f"{resolved}.observations.series[]"
    series_refs = [
        ref
        for series in spec.read("resolved.observations.series", ctx) or []
        for ref in _refs(
            read_path(series, "entities", ctx, at=series_at), "ref", f"{series_at}.entities[]", ctx
        )
    ]
    if preview_complete and series_refs:
        return list(dict.fromkeys(series_refs))
    planned = _refs(spec.read("resolved.entities", ctx), "ref", f"{resolved}.entities[]", ctx)
    if spec.read("resolved.scope.axis", ctx) == "observationAbout":
        members = spec.read("resolved.scope.members", ctx)
        planned += _refs(members, "ref", f"{resolved}.scope.members[]", ctx)
    if not set(series_refs) <= set(planned):
        return None
    return list(dict.fromkeys(planned))


def _coverage_window(spec: SpecView, ctx: ScoreContext) -> TimeWindow | None:
    start = spec.read("resolved.coverage.window.start_year", ctx)
    end = spec.read("resolved.coverage.window.end_year", ctx)
    if start is None and end is None:
        return None
    return TimeWindow(start_year=start, end_year=end)


def _exact(spec: SpecView, response: Mapping[str, Any], ctx: ScoreContext) -> Result:
    stated = _stated_facets(spec, ctx)
    warning_codes = _spec_warning_codes(response, spec.read("spec_id", ctx), ctx)
    unit = stated.get("unit")
    if unit is not None and unit[1] and CURRENCY_UNMAPPED_OMITTED in warning_codes:
        return _unchecked(
            "conversion_omitted", "the count leaves out the rows the conversion omitted"
        )
    resolved = f"{spec.at}.resolved"
    try:
        served = served_stat_vars(spec, ctx)
    except MembersRefused as exc:
        return fail(str(exc))
    except READ_ERRORS as exc:
        return infra_error(f"members door: {type(exc).__name__}: {exc}")
    if served is None:
        return infra_error("an exact count reads the served StatVars; no members door")
    stat_vars = list(dict.fromkeys(served))
    latest = spec.read("definition.identity.requested_window.kind", ctx) == "latest"
    window = _coverage_window(spec, ctx)
    expected_count = spec.read("resolved.coverage.observation_count", ctx)
    entities_at = f"{resolved}.entities[]"
    entities = [
        (read_path(e, "ref", ctx, at=entities_at), read_path(e, "role", ctx, at=entities_at))
        for e in spec.read("resolved.entities", ctx) or []
    ]

    if any(role in _COVERAGE_ROLE_FOR_AXIS for _ref, role in entities):
        if ctx.coverage is None:
            return infra_error(
                "a role-bearing answer is counted from coverage; no coverage service"
            )
        try:
            snapshot = ctx.coverage.snapshot()
        except READ_ERRORS as exc:
            return infra_error(f"coverage snapshot: {type(exc).__name__}: {exc}")
        stamp = spec.read("resolved.coverage_stamp", ctx)
        if stamp is not None and stamp != snapshot.stamp:
            return _unchecked(
                "coverage_stamp_mismatch",
                f"answer stamp {stamp!r}, eval stamp {snapshot.stamp!r}",
            )
        if not all(role in _COVERAGE_ROLE_FOR_AXIS for _ref, role in entities):
            return _unchecked("mixed_entity_roles", "observationAbout beside a flow role")
        bound: dict[CoverageRole, frozenset[str]] = {}
        for ref, role in entities:
            if role != "observationAbout":
                coverage_role = _COVERAGE_ROLE_FOR_AXIS[role]
                bound[coverage_role] = bound.get(coverage_role, frozenset()) | {ref}
        facets_by_series = _role_series_facets(stat_vars, bound, snapshot)
        # The rows a conversion leaves out of the election make the answer unchecked above.
        elected = (
            snapshot.index.elected_years(
                variables=stat_vars, bound=bound, admits=_stated_admits(stated, snapshot)
            )
            if latest
            else {}
        )
        # Coverage holding no row of the bound tuples reaches a total of 0 alone.
        totals = (
            _reachable_totals(
                facets_by_series, stated, window=window, latest=latest, elected=elected
            )
            or 1
        )
        return _count_result(totals, expected_count)

    places = _served_places(spec, ctx, preview_complete=not warning_codes & _PARTIAL_PREVIEW_CODES)
    if places is None:
        return _unchecked(
            "partial_preview",
            "the partial preview's series name places outside its entities and scope",
        )
    if not places:
        return _unchecked("no_places", "the answer names no entity, series place or scope member")
    if ctx.graph is None:
        return infra_error("materialisation needs a graph client")
    try:
        by_sv = ctx.graph.observation_facets_batch(stat_vars, places, needs_dates=True)
    except READ_ERRORS as exc:
        return infra_error(f"graph observation_facets_batch: {type(exc).__name__}: {exc}")
    totals = _reachable_totals(
        [(sv, by_sv.get(sv, {}).get(place, [])) for sv in stat_vars for place in places],
        stated,
        window=window,
        latest=latest,
        elected={},
    )
    if totals is None:
        return _unchecked("no_records", f"the graph holds no facet at {len(places)} place(s)")
    return _count_result(totals, expected_count)


def _part(
    block: StatusExpect, part: PartView, response: Mapping[str, Any], ctx: ScoreContext
) -> Result:
    spec = SpecView(
        f"{PartView.path}.spec",
        f"/parts/{part.index}/spec",
        read_path(part.body, "spec", ctx, at=PartView.path),
    )
    kind = spec.read("resolved.coverage.kind", ctx)
    if spec.read("resolved.coverage.has_data", ctx) is not True:
        return fail("coverage.has_data is not true")
    match kind:
        case "bare":
            return PASS
        case "breadth":
            at = f"{spec.at}.resolved.coverage.dimensions[]"
            dimensions = spec.read("resolved.coverage.dimensions", ctx) or []
            empty = [
                read_path(d, "label", ctx, at=at)
                for d in dimensions
                if (read_path(d, "count", ctx, at=at) or 0) <= 0
            ]
            return fail(f"breadth dimensions with count<=0: {empty}") if empty else PASS
        case "exact":
            return _exact(spec, response, ctx)
    return fail(f"unknown coverage kind {kind!r}")


@register("materialisation", modal_only=True)
def materialisation(case: Case, response: Mapping[str, Any], ctx: ScoreContext) -> Result:
    """Every definite part's coverage holds in the graph or the coverage index. A pass or
    n/a keeps each part's count or reason in its detail. A ``definite`` or ``mixed`` block
    asserts the check (``cases.assertion_checks``), so a part pairing with no block reads
    unreached."""
    expected = expected_parts(case, response, ctx)
    results = [
        (part, run_part(_part, block, part, response, ctx))
        for block, part in part_pairs(expected)
        if part.status == "definite"
    ]
    missed = unreached(case, expected, check="materialisation")
    combined = by_part(case, ((part.index, result) for part, result in results), missed)
    if combined.verdict not in (Verdict.PASS, Verdict.NA):
        return combined
    details = [f"parts[{part.index}]: {r.detail}" for part, r in results if r.detail]
    return Result(combined.verdict, "; ".join(details))
