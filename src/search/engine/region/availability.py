"""The availability read: whether each admissible reading of a part has data where the
question asks, D(x) under the status rule (`region.status_rule`).

A part that names places reads D at them: a subject-only reading from the observations at
the named countries, the nodes of the sets the part names and their members, within the
requested years where the probe states them; a role-bearing reading from the coverage
snapshot, on the roles the places bind (`series_at_places`). A part that names no place
reads D in the default scope each subject-only reading declares, from the default-scope
presence read (`place.read_default_presence`): data anywhere in that scope. A role-bearing
reading with no place leaves D unknown, and its own read decides.

Derive reads the named countries for every recalled StatVar (`ShapeDraft.place_available`),
so a part naming only countries makes no further read for D. A part naming a set makes one
more at every place the part names (`read_menu_availability`), over the menu's subject-only
StatVars. That read covers every member of every set: a sample can show data present but
never absent. The request deadline and a response byte cap bound it.
Each read stamps the places it covered on the shape (`ShapeDraft.availability_places`), and
a read that found no data settles D as absent only when it covered every place the part
names (`asked_places`). A failed read, a slow or oversized one among them, leaves D unknown,
which is no evidence either way, and so does a read of fewer places than the part names.

One more read settles no D. Where `max_candidates` cuts the breakdowns of a family with no
total, the family's members recall did not return are read at the shape's
`availability_places`, countries alone included, so the cut ranks each breakdown by its
data in the years asked for (`phase_place_materialise._breakdown_coverage`). That read
stamps nothing on the shape, and a part naming no place makes none.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Collection, Mapping, Sequence
from typing import Protocol

from materialise.place_scope import DefaultPresence
from search.engine.clients.graph import EngineGraphClient, Facet
from search.engine.config import settings
from search.engine.degraded_reads import note_degraded_read
from search.engine.extract import DateRequest
from search.engine.reading_coverage import ReadingCoverage, holds_in, observed_years
from search.engine.region.coverage_series import holds_series, requested_window
from search.engine.region.place_roles import PlaceAssignment, bind_place_roles, names_a_place
from search.engine.region.place_set import PlaceSet
from search.engine.region.state import ResolveState
from search.engine.retrieval.bound_values import index_carries_shape
from search.engine.retrieval.standard import _facet_reads
from search.engine.shape import ShapeDraft, is_role_bearing
from search.models import TimeWindow

logger = logging.getLogger(__name__)

type _ShapeGroupKey = tuple[str, str, str, str, str | None, str | None, bool, frozenset[str]]


def _shape_group_key(shape: ShapeDraft) -> _ShapeGroupKey:
    """Return a collision-free key for one derived shape group. Each copy of a shape
    narrowed to another breakdown (`ShapeDraft.omitted_slots`) is a group of its own."""
    return (
        shape.shape_id,
        shape.pop_type_dcid,
        shape.meas_prop_dcid,
        shape.stat_type_dcid,
        shape.meas_qual_dcid,
        shape.meas_denom_dcid,
        shape.meas_denom_unresolved,
        shape.omitted_slots,
    )


def asked_places(place_sets: Mapping[str, PlaceSet]) -> list[str]:
    """Every place the part names: each set's node, then its members, without repeats."""
    places: list[str] = []
    for place_set in place_sets.values():
        if place_set.node is not None:
            places.append(place_set.node.dcid)
        places.extend(place_set.members or ())
    return list(dict.fromkeys(places))


def menu_read_places(
    place_sets: Mapping[str, PlaceSet], derive_places: Sequence[str] | None
) -> list[str]:
    """The places the menu's availability read covers, empty when derive's probe at
    `derive_places`, the named countries, already read every place the part names. Else
    every place the part names (`asked_places`)."""
    places = asked_places(place_sets)
    return [] if set(places) <= set(derive_places or ()) else places


def read_menu_availability(
    shapes: Sequence[ShapeDraft],
    *,
    graph: EngineGraphClient,
    places: Sequence[str],
    deadline: float | None,
) -> tuple[list[ShapeDraft], bool]:
    """`shapes` with each subject-only one's coverage at `places`
    (`ShapeDraft.named_place_coverage`) and those places (`ShapeDraft.availability_places`),
    from one batched facet read over their StatVars, and whether that read failed.

    The read is bounded by `deadline`, the request's monotonic deadline, and by the
    response byte cap (`sdmx_max_bytes`). A failed read, a slow or oversized one among
    them, is noted as lost (`degraded_reads.note_degraded_read`) and leaves every shape as
    it was. Derive's read covers fewer places than `places`, so an absence it found settles
    nothing there (`_subject_has_data`) and D stays unknown. Role-bearing shapes are left as
    they are: the subject facets this read sees do not hold their observations. Derive's
    availability at the named countries (`ShapeDraft.place_available`), which the slot
    defaults read, stays as derive read it."""
    subject = [shape for shape in shapes if not is_role_bearing(shape)]
    stat_vars = list(dict.fromkeys(sv for shape in subject for sv in (shape.sv_arc_facts or {})))
    if not stat_vars or not places:
        return list(shapes), False
    coverage = read_coverage(
        stat_vars, graph=graph, places=places, deadline=deadline, read="menu availability"
    )
    if coverage is None:
        return list(shapes), True
    read: list[ShapeDraft] = []
    for shape in shapes:
        if is_role_bearing(shape):
            read.append(shape)
            continue
        at = {sv: coverage.get(sv, {}) for sv in (shape.sv_arc_facts or {})}
        read.append(
            dataclasses.replace(
                shape, named_place_coverage=at, availability_places=frozenset(places)
            )
        )
    return read, False


def read_coverage(
    stat_vars: Sequence[str],
    *,
    graph: EngineGraphClient,
    places: Sequence[str],
    deadline: float | None,
    read: str,
) -> dict[str, ReadingCoverage] | None:
    """The places among `places` at which each of `stat_vars` has observations, each with
    the years its facets there hold (`reading_coverage.observed_years`), from batched facet
    reads in the chunks standard retrieval reads (`standard._facet_reads`), each bounded by
    `deadline` and the response byte cap (`sdmx_max_bytes`). `None` when a read fails, a
    slow or oversized one among them, noted as the lost `read`
    (`degraded_reads.note_degraded_read`)."""
    probe: dict[str, dict[str, list[Facet]]] = {}
    try:
        for chunk in _facet_reads(stat_vars, places):
            probe.update(
                graph.observation_facets_batch(
                    chunk,
                    list(places),
                    False,
                    max_bytes=settings().sdmx_max_bytes,
                    deadline=deadline,
                )
            )
    except Exception:
        logger.warning(
            "%s read failed for %d StatVar(s) at %d place(s); their availability stays unknown",
            read,
            len(stat_vars),
            len(places),
            exc_info=True,
        )
        note_degraded_read(read)
        return None
    return {
        sv: {
            place: years
            for place, facets in by_place.items()
            if (years := observed_years(facets)) is not None
        }
        for sv, by_place in probe.items()
    }


class SeriesReader(Protocol):
    """Reads whether a role-bearing shape holds a series at the part's places
    (`series_at_places`)."""

    def __call__(
        self, shape: ShapeDraft, assignment: PlaceAssignment | None = None, /
    ) -> bool | None: ...


def series_at_places(state: ResolveState) -> SeriesReader:
    """Whether each role-bearing shape's family holds a series at the places the part
    names, on the roles an assignment binds them to, else on those `bind_place_roles`
    gives (`coverage_series.holds_series`). `None` when the shape is subject-only, refuses
    the places, binds none, or has no member the coverage index holds."""

    def read(shape: ShapeDraft, assignment: PlaceAssignment | None = None) -> bool | None:
        if not is_role_bearing(shape):
            return None
        bound_roles = bind_place_roles(state, shape) if assignment is None else assignment
        if isinstance(bound_roles, str) or bound_roles.binds_no_place:
            return None
        return holds_series(state, shape, bound_roles.sets)

    return read


def _in_window(coverage: Mapping[str, ReadingCoverage], window: TimeWindow) -> bool:
    """Whether some member holds a point at some place inside `window`, or may: a span the
    probe states without its years counts when it overlaps the window."""
    return any(holds_in(by_place, window) for by_place in coverage.values())


def _subject_has_data(
    shape: ShapeDraft, date_request: DateRequest | None, *, asked: Collection[str]
) -> bool | None:
    """D at `asked`, the places the part names, for a subject-only shape: whether a member
    holds data at a place the probe read, inside the requested years when the probe states
    them.

    The menu read over every place the part names (`ShapeDraft.named_place_coverage`)
    settles D where it ran, else derive's read at the named countries. Both reads cover the
    shape's recalled members and state each facet's first and last years. A read that found
    nothing, or nothing inside the requested years, settles D as absent only where it
    covered every asked place (`ShapeDraft.availability_places`); else D stays unknown. So
    does a read that found nothing for a shape the code index carries, whose family holds
    members recall did not return."""
    if shape.named_place_coverage is not None:
        coverage: Mapping[str, ReadingCoverage] | None = shape.named_place_coverage
        available: bool | None = any(coverage.values())
    else:
        coverage = shape.member_coverage
        available = shape.place_available
    covered = shape.availability_places.issuperset(asked)
    if available is False and (not covered or index_carries_shape(shape)):
        return None
    if not available:
        return available
    window = requested_window(date_request)
    if window is None or coverage is None or _in_window(coverage, window):
        return True
    return False if covered else None


def _default_scope_has_data(shape: ShapeDraft, presence: DefaultPresence | None) -> bool | None:
    """D in the default scope a subject-only shape declares: whether a member's default
    scope holds a place. Unknown when the presence read does not cover every member."""
    members = list(shape.sv_arc_facts or ())
    if presence is None or not members or any(sv not in presence.scopes for sv in members):
        return None
    return any(presence.scopes[sv].places for sv in members)


def has_data_where_asked(
    state: ResolveState,
    shape: ShapeDraft,
    *,
    series: SeriesReader | None = None,
    assignment: PlaceAssignment | None = None,
) -> bool | None:
    """D(x) for `shape`: whether it has data where the part asks, `None` when no read
    settled it. `series` is the part's coverage reader (`series_at_places`), built once per
    caller, and `assignment` the role assignment to read a role-bearing shape on, else the
    one `bind_place_roles` gives."""
    if names_a_place(state):
        if is_role_bearing(shape):
            read = series if series is not None else series_at_places(state)
            return read(shape, assignment)
        return _subject_has_data(shape, state.date_request, asked=asked_places(state.place_sets))
    if is_role_bearing(shape):
        return None
    return _default_scope_has_data(shape, state.default_presence)
