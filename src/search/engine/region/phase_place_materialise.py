"""Assign place sets to roles, settle them, and materialise the elected shape.

Handle no-data results after earlier phases fall through.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import logging
import threading
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final, Literal, cast

import dc_graph
from dfo_platform.executor import submit_with_context
from dfo_platform.trace import get_trace_id
from materialise.place_scope import DefaultPresence
from materialise.roles import RoleAxis, axis_for_role
from search.engine.bind import SlotBindingDraft, sole_option_properties
from search.engine.clients.graph import EngineGraphClient, force_full_facets
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.config import SEARCH_MAX_UNIT_ALTERNATIVES, settings
from search.engine.departures import direction_inferred
from search.engine.envelope import _emit_step, now_ms
from search.engine.errors import FamilyNotResolvableError, GraphInfraError, RoleSignatureError
from search.engine.interpret import PinnedResolved, Recall
from search.engine.place import earth_countries, read_default_presence
from search.engine.quantity import (
    NamedDenominator,
    check_denominator,
    schema_denominator_kind,
    serves_quantity,
    stated_denominator,
)
from search.engine.reading_coverage import ReadingCoverage, holds_in
from search.engine.region.alternate_guard import AlternateJudge, query_pins
from search.engine.region.availability import (
    _shape_group_key,
    has_data_where_asked,
    read_coverage,
    series_at_places,
)
from search.engine.region.candidate_options import _elected_sibling_warning
from search.engine.region.candidates import (
    _Alternate,
    _AlternatesRead,
    _materialise_alt_shapes,
    alternate_representative,
    settle_alternate,
    walk_alternate,
)
from search.engine.region.codes import (
    DENOMINATOR_UNVERIFIED,
    MEMBERSHIP_UNVERIFIED,
    SEAM_OFF_INFO_CODE,
    _elect_no_data_reason,
)
from search.engine.region.coverage_series import requested_window
from search.engine.region.decisions import (
    PlaceSetNotAdmittedError,
    PlaceSetRefusedError,
    PlaceSetTooLargeError,
)
from search.engine.region.grounding import (
    _suggest_nearest_real,
    membership_unverified_warning,
    memoised_labels,
)
from search.engine.region.materialise_step import (
    NoDataMaterialise,
    UnsupportedMaterialise,
    interpret_materialised,
    run_materialise,
)
from search.engine.region.observations.dispatch import _attach_observations
from search.engine.region.observations.entities import _observation_entities
from search.engine.region.observations.fetch import _build_observations
from search.engine.region.phase_election import (
    _settle_served_dataset,
    narrowed_to_breakdown,
    sampled_claims,
    without_sampled_claims,
)
from search.engine.region.phase_intake import not_admitted_entity
from search.engine.region.phase_rank_bind import _query_records
from search.engine.region.phase_shared import (
    _elected_flagged_match,
    _role_incompatible_no_data,
    _role_not_resolved_no_data,
    _unsupported,
)
from search.engine.region.place_context import (
    PlaceContext,
    UnsettledPlace,
    cannot_take_default_scope,
    default_scope_stat_vars,
    install_place_warnings,
    settle_place,
    settle_sets,
    settled_shape,
    stamp_offered_place_warnings,
    takes_default_scope,
    with_where,
)
from search.engine.region.place_roles import (
    PlaceAssignment,
    PlaceRoleRefusal,
    bind_alternate_place_roles,
    bound_after_election,
    direction_assignments,
    first_spellings,
    holds_whole_total,
    names_a_place,
    request_named_groups,
    undirected_place_name,
    whole_total_candidates,
)
from search.engine.region.requirements import record_reading, settle_answer, stray_values
from search.engine.region.result import FacetAxisFilter, RegionResult
from search.engine.region.state import (
    AlternateUnevaluatedReason,
    ResolveState,
    TakerStatVars,
    for_shape,
    presence_covering,
    taker_stat_vars,
)
from search.engine.region.status_rule import (
    Alternative,
    MenuReading,
    OpenAxes,
    StatusPlan,
    open_axes,
    plan_status,
    shares_role_class,
)
from search.engine.region.unrealisable import unbind_unrealisable
from search.engine.retrieval.bound_values import (
    RoleKeys,
    bound_values_from_bindings,
    elected_role_class_shape,
    entity_role_keys,
    narrowed_to,
    shape_population_arcs,
    shape_query,
    sole_shared_value,
    without_values,
)
from search.engine.retrieval.facts import (
    read_constraints,
    read_declared_roles,
)
from search.engine.retrieval.retrieve import (
    BoundValuesUnrealisable,
    Materialised,
    MaterialisedCandidates,
    NoDataDraft,
    UnsupportedDraft,
)
from search.engine.retrieval.role_resolve import unrealisable_bound_values
from search.engine.retrieval.standard import Candidates, FacetBatch, Narrow
from search.engine.shape import (
    ShapeDraft,
    is_role_bearing,
)
from search.engine.unit.base_year import declared_base_year
from search.engine.unit.bind import UnitDefaultPolicy, bind_unit_text
from search.engine.unit.refs import satisfies as _unit_satisfies
from search.models import (
    CandidateBasis,
    FacetBasis,
    IdentityUnitRequest,
    NearestRealSpec,
    NoDataDenominator,
    Relaxation,
    TimeWindow,
    UnitAlternative,
    Warning,
)

logger = logging.getLogger(__name__)

_ALTERNATES_WORKERS: Final[int] = 8
"""Threads that make, process-wide, the optional reads candidate admission's alternates need
ahead of it, beside the shape in play's own reads (`_submit_alternates_read`): the listing
of Earth's countries and the presence read a query naming no place needs beyond the shape
in play's. No member walk runs here (`_WALK_POOL`), so these reads keep every worker. A
read holds one while its source answers, up to `SEARCH_GRAPH_TIMEOUT_S` per attempt. A read
that finds all of them busy is not made, so none waits in a queue or holds a worker of the
default executor: a presence read not made leaves its alternates unevaluated
(`_read_alternates_presence`). A read still running when the request ends finishes on its
worker, and nothing waits for it."""

_ALTERNATES_POOL: Final = ThreadPoolExecutor(
    max_workers=_ALTERNATES_WORKERS, thread_name_prefix="alternates-reads"
)
_ALTERNATES_SLOTS: Final = threading.BoundedSemaphore(_ALTERNATES_WORKERS)

# Sized from the walks a query part starts: the 73 first-answer scenarios
# (`tests/engine/test_first_answer_place_scenarios.py`) start at most 3 and 1.1 on average,
# and a request runs at most `SEARCH_MAX_VARIABLE_CONCURRENCY` (8) parts at once, so 16
# workers run every walk of about 14 parts (16 / 1.1) with none queued, which is less than
# two requests at their full fan-out. A walk holds its worker for its three or so graph
# reads.
_WALK_WORKERS: Final[int] = 16
"""Threads that run, process-wide, the member walks of the alternates and re-election
siblings a request may offer (`_start_alternate_walks`). Every walk is queued here and none
is refused: admission waits for a walk still queued, so a busy pool delays a request's
alternates and leaves none unevaluated. A walk still queued when its pass ends is
cancelled, and a running one finishes with its result dropped
(`pipeline._cancel_queued_reads`). No walk runs on the default executor or on the presence
reads' pool (`_ALTERNATES_POOL`)."""

_WALK_POOL: Final = ThreadPoolExecutor(
    max_workers=_WALK_WORKERS, thread_name_prefix="alternate-walks"
)


def _note_named_places(state: ResolveState) -> None:
    """Record which named places the caller pinned, and warn when the place seam is off."""
    # The type checker cannot narrow ``state.rcl`` from an earlier phase.
    rcl = cast(Recall, state.rcl)

    # Subject shapes bind all places in ``where``. Role-bearing shapes bind destinations
    # there and carry origins in ``identity.entities``. Arc-derived bindings describe only
    # alternate shapes, which have no bind call.
    if not state.pac:
        state.warnings.append(
            Warning(
                code=SEAM_OFF_INFO_CODE,
                severity="info",
                message="place_as_constraint is OFF; all entities treated as subjects.",
            )
        )

    # Pins prove existence. Graph labels supply their display text. A place takes the
    # text of its first spelling in the query (`place_roles.place_texts`), so it is pinned
    # when that spelling is a pin.
    state.pinned_entity_dcids = frozenset(
        dcid
        for dcid, name in first_spellings(state.entities, state.place_sets).items()
        if isinstance(rcl.resolutions.get(name), PinnedResolved)
    )


async def _phase_place_bindings(state: ResolveState) -> RegionResult | None:
    """Assign the elected shape's place sets to its roles and install it with the place
    context it has before it is read.

    Role refusals return their terminal result. Every alternate's places are assigned here
    too, once (`_bind_alternates`), the siblings zero-observation re-election would try are
    chosen (`_reelection_siblings`), and the admission guards and candidate budget are
    applied (`_plan_admission`), all before any read.

    A place set that holds a node and members settles on the shape's own read
    (`_run_materialise_task`), and until then the context holds its members
    (`UnsettledPlace.unread`). When no read decides a set, the context is the settled one,
    and places over the bound on the places a Spec binds answer `entity_set_too_large`.

    A subject-only shape the query names no place for takes the default scope of its own
    StatVars. The presence read that scope comes from is the request's one read on this
    path: it also covers every shape the request may go on to probe over a default scope,
    the alternates admission would materialise, the re-election siblings and the shapes a
    widened search would add, so each derives its scope from it and no read is started for
    a shape the answer may not reach. A role-bearing elected shape makes no presence read;
    the read its subject-only alternates need starts here and runs beside its
    materialisation (`_start_alternates_presence_read`). An elected shape that cannot take
    a default scope refuses the request (`DefaultScopeIndexSourcedError`).

    A role-bearing family the walk or the catalogue's tables read whole and found no member
    of under the bindings (`MembershipResolution.no_member`) has bound values no member holds
    together (`role_resolve.unrealisable_bound_values`). They lose a minimal set here, by
    the rule the read applies (`unrealisable.unbind_unrealisable`), and the family is
    settled again, so the place context is built once, on the family of the bindings the
    answer states. `membership_unverified` then says only that a family could not be read.

    The walk reads each slot a recall sample alone claimed open
    (`phase_election.without_sampled_claims`), so the family it settles holds the members
    the claim would exclude, and the claim is confirmed or refuted on that family
    (`_judge_sampled_claims`). A family no walk settles leaves the claim standing on the
    sample, under `membership_unverified`.
    """
    _note_named_places(state)

    # Binding repair bound the elected shape's places (`ResolveState.query_place_roles`).
    bound = state.query_place_roles
    assert bound is not None, "binding repair binds the query's places"
    if bound == "role_not_resolved":
        bound = _settle_direction(state)
    if bound == "role_incompatible":
        return await _role_incompatible_no_data(state)
    if bound == "role_not_resolved":
        return await _role_not_resolved_no_data(state)
    if bound == "no_direction_holds":
        return await _role_not_resolved_no_data(state, offer=False)
    state.elected_place_roles = bound
    await _settle_breakdown(state)
    # The bound roles elect one role class among the shape's members, and every later
    # step reads the shape narrowed to it. The election reads a held family whole, so it
    # runs off the event loop. The mint's walk then settles the members, off the event loop
    # too, for any shape the index or the catalogue's tables leave unsettled and with a
    # member graph to walk. Members it does not settle still answer, under a warning.
    role_keys = bound.role_keys
    role_query = shape_query(
        state.shape_draft,
        without_sampled_claims(state.bindings),
        role_keys.bound,
        open_role_keys=role_keys.open,
    )
    recalled = state.shape_draft
    try:
        resolution = await asyncio.to_thread(
            elected_role_class_shape, state.shape_draft, role_query
        )
    except (RoleSignatureError, FamilyNotResolvableError) as exc:
        return _election_refusal(state, exc)
    state.shape_draft = resolution.value
    # The alternates and siblings start their walks now, beside the elected shape's own, under
    # the plan the recalled election gives. A walk that changes the elected shape makes the
    # plan again, and a walk the new plan needs and this one did not starts then.
    planned, plan = state.shape_draft, _plan_alternates(state)
    if not names_a_place(state):
        _start_countries_listing(state, plan)
    if state.role_class_walks is not None and not resolution.authority.settled:
        try:
            resolution = await asyncio.to_thread(
                elected_role_class_shape,
                state.shape_draft,
                role_query,
                walks=state.role_class_walks,
                arcs_graph=state.graph,
                deadline=state.deadline,
            )
        except (RoleSignatureError, FamilyNotResolvableError) as exc:
            return _election_refusal(state, exc)
        state.shape_draft = resolution.value
    # A role-bearing family read whole that holds no member under the bindings is not one
    # the walk failed to read: bound values that match no member together lose a set
    # (`unbind_unrealisable`) and the family is settled again under what remains, so the
    # members every later step reads are the family of the bindings the answer states. The
    # conflicts are the mint's (`role_resolve._mint_conflicts`), which the role path reads
    # by; a standard part relaxes at its own read (`standard._admit_standard_members`).
    relaxed = False
    while (
        resolution.no_member
        and is_role_bearing(state.shape_draft)
        and (
            unrealisable := await asyncio.to_thread(
                unrealisable_bound_values, state.shape_draft, role_query, state.bindings
            )
        )
        is not None
    ):
        state.shape_draft = recalled
        if (refused := await _unbind(state, unrealisable, role_keys)) is not None:
            return refused
        recalled = state.shape_draft
        relaxed = True
        role_query = shape_query(
            recalled,
            without_sampled_claims(state.bindings),
            role_keys.bound,
            open_role_keys=role_keys.open,
        )
        try:
            resolution = await asyncio.to_thread(
                elected_role_class_shape,
                recalled,
                role_query,
                walks=state.role_class_walks,
                arcs_graph=state.graph,
                deadline=state.deadline,
            )
        except (RoleSignatureError, FamilyNotResolvableError) as exc:
            return _election_refusal(state, exc)
        state.shape_draft = resolution.value
    state.walked = resolution.walked
    if refuted := _judge_sampled_claims(state, role_keys):
        if (refused := await _unbind_refuted_claims(state, refuted, role_keys)) is not None:
            return refused
        relaxed = True
    if not resolution.authority.settled:
        state.warnings.append(membership_unverified_warning())
    # The place context below is built for the shape the election returned.
    bound = bound_after_election(state, recalled, bound, state.shape_draft)
    if bound == "role_incompatible":
        return await _role_incompatible_no_data(state)
    if bound == "role_not_resolved":
        return await _role_not_resolved_no_data(state)
    if relaxed or state.shape_draft is not planned:
        plan = _plan_alternates(state)
    (unsettled,) = await _place_contexts(
        state,
        [(state.shape_draft, state.bindings, bound)],
        group=[*plan.alt_indices, *state.reelection_siblings],
    )
    try:
        place = unsettled.unread()
    except PlaceSetTooLargeError as exc:
        return _too_many_places(state, exc)
    except PlaceSetNotAdmittedError as exc:
        return _not_admitted_places(state, exc)
    _install_shape(
        state, state.shape_draft, state.bindings, place, unsettled, index=state.elected_index
    )
    if place.default_scope is None:
        state.alternates_presence = _start_alternates_presence_read(state, plan)
    return None


def _election_refusal(
    state: ResolveState, exc: RoleSignatureError | FamilyNotResolvableError
) -> RegionResult:
    """The refusal of an elected shape whose members the mint refuses: role sets its walk
    cannot order (`RoleSignatureError`), or a member at fault, so its series are not
    resolved (`FamilyNotResolvableError`)."""
    if isinstance(exc, RoleSignatureError):
        return state.no_data("role_signature_incompatible", llm_usage=state.var_usage)
    return state.no_data("variable_not_resolved", llm_usage=state.var_usage)


def _settle_direction(
    state: ResolveState,
) -> PlaceAssignment | PlaceRoleRefusal | Literal["no_direction_holds"]:
    """The elected shape's places when one is named without a direction no rule places
    (`place_roles.direction_assignments`): the availability read runs once per role
    assignment (`availability.series_at_places`). Exactly one with data is served, and the
    answer states the direction it reads the place on (`departures.direction_inferred`).
    When several hold data the first is served and each other is offered beside it as a
    `direction` candidate: a copy of the elected shape on the menu, bound on its own
    assignment (`ResolveState.direction_rivals`). When readings exist and every read says
    empty (`False`) the places refuse ``no_direction_holds``, a ``role_not_resolved`` that
    offers no reading, since each is known empty. A read the coverage index cannot answer
    (`None`) is unknown. Otherwise they refuse ``role_not_resolved``, as a question
    whose direction no data settles. The served assignment binds the first whole total a
    side it leaves unnamed offers where the bindings hold its series
    (`place_roles.holds_whole_total`); each rival decides its own when it is walked
    (`_walked_alternates`)."""
    shape = state.shape_draft
    series = series_at_places(state)
    reads = [
        (axis, assignment, series(shape, assignment))
        for axis, assignment in direction_assignments(state, shape)
    ]
    if reads and all(read is False for *_, read in reads):
        return "no_direction_holds"
    held = [(axis, assignment) for axis, assignment, read in reads if read]
    name = undirected_place_name(state)
    if not held or name is None:
        return "role_not_resolved"
    (axis, served), *others = held
    if not others:
        state.inferred_facts.append(direction_inferred(name, axis))
    for _axis, assignment in others:
        state.direction_rivals[len(state.capped_shapes)] = assignment
        state.capped_shapes.append(shape)
    # Binding repair settled the bindings under no role keys, so each whole total the
    # served direction offers is judged under them, in the order offered
    # (`place_roles.whole_total_candidates`). A claim selects no series until the walk
    # confirms it, so the total is judged on the claim-free bindings.
    held_bindings = without_sampled_claims(state.bindings)
    return next(
        (
            whole
            for whole in whole_total_candidates(state, shape, served)
            if holds_whole_total(state, shape, whole, held_bindings)
        ),
        served,
    )


async def _settle_breakdown(state: ResolveState) -> None:
    """The elected shape's breakdown when its family holds no total and the question names
    none that serves a member (`standard.Candidates`): one copy of the shape per breakdown
    signature, each narrowed to the bound properties and that signature
    (`phase_election.narrowed_to_breakdown`) and bound with the signature's slots open
    (`_breakdown_bindings`). The copies are ordered by `_breakdown_rank` and capped at
    `max_candidates`, a cut the candidate set states (`ResolveState.breakdown_truncated`,
    `CandidateSet.truncated`). The first is installed as the elected shape; each other is
    offered beside it as a `breakdown` candidate (`ResolveState.breakdown_rivals`). A lone
    copy answers alone, narrower than asked (`standard.Narrow.lone_signature`)."""
    decision = state.breakdown
    if not isinstance(decision, Candidates):
        return
    shape = state.shape_draft
    coverage = await _breakdown_coverage(state, decision)
    rank = _breakdown_rank(decision, shape, coverage, requested_window(state.date_request))
    signatures = sorted(decision.signatures, key=rank)[: settings().max_candidates]
    state.breakdown_truncated = len(signatures) < len(decision.signatures)
    copies = [
        (
            narrowed_to_breakdown(shape, decision.bound | signature),
            _breakdown_bindings(state.bindings, shape, signature),
        )
        for signature in signatures
    ]
    (elected, bindings), *others = copies
    state.shape_draft = state.capped_shapes[state.elected_index] = elected
    state.bindings = bindings
    if not others:
        state.breakdown = Narrow(decision.bound | signatures[0], lone_signature=True)
    for copy, copy_bindings in others:
        state.breakdown_rivals[len(state.capped_shapes)] = copy_bindings
        state.capped_shapes.append(copy)


async def _breakdown_coverage(
    state: ResolveState, decision: Candidates
) -> Mapping[str, ReadingCoverage]:
    """The coverage `_breakdown_rank` reads for the members of `decision`: the elected
    shape's at the places the part names (`ShapeDraft.named_place_coverage`, else
    `ShapeDraft.member_coverage`). When `max_candidates` cuts a signature, the members
    recall did not return, which the catalogue's tables or the code index hold in the
    family, are read at the same places (`ShapeDraft.availability_places`) in batched facet
    reads (`availability.read_coverage`), so a breakdown holding data only in them is
    ranked as one recalled would be. The read takes at most `region_max_members` of them,
    the bound the facet prewarm holds, one from each breakdown in turn, so a family of
    thousands reads a sample of each breakdown. A failed read leaves them without data. A
    part naming no place has no coverage at places and makes no read, so its cut ranks on
    the members recall returned."""
    shape = state.shape_draft
    coverage = shape.named_place_coverage or shape.member_coverage
    if coverage is None:
        return {}
    by_breakdown = [
        [sv for sv in members if sv not in coverage] for members in decision.members.values()
    ]
    unread = [sv for turn in itertools.zip_longest(*by_breakdown) for sv in turn if sv is not None][
        : settings().region_max_members
    ]
    if (
        len(decision.members) <= settings().max_candidates
        or not unread
        or not shape.availability_places
    ):
        return coverage
    read = await asyncio.to_thread(
        read_coverage,
        unread,
        graph=state.graph,
        places=sorted(shape.availability_places),
        deadline=state.deadline,
        read="breakdown availability",
    )
    return {**coverage, **(read or {})}


def _breakdown_rank(
    decision: Candidates,
    shape: ShapeDraft,
    coverage: Mapping[str, ReadingCoverage],
    window: TimeWindow | None,
) -> Callable[[frozenset[str]], tuple[int, int]]:
    """The sort key of a breakdown signature of `decision` over `shape`: its members with
    data in `coverage` inside `window`, the years the part asks for
    (`reading_coverage.holds_in`), then its recalled members (`ShapeDraft.scored_members`),
    most first."""

    def rank(signature: frozenset[str]) -> tuple[int, int]:
        members = decision.members[signature]
        with_data = sum(holds_in(coverage.get(sv) or {}, window) for sv in members)
        recalled = sum(sv in shape.scored_members for sv in members)
        return -with_data, -recalled

    return rank


def _breakdown_bindings(
    bindings: Sequence[SlotBindingDraft], shape: ShapeDraft, signature: frozenset[str]
) -> list[SlotBindingDraft]:
    """`bindings` for the copy of `shape` narrowed to `signature`: each slot of the
    signature open, as a binding already open or ``unbound``, and every other what or how
    slot bound to a value or left out."""
    by_property = {b.property_dcid: b for b in bindings if b.property_dcid is not None}
    kept = [
        b
        for b in bindings
        if b.axis not in ("what", "how")
        or b.property_dcid is None
        or (b.kind == "unbound" if b.property_dcid in signature else b.kind in ("value", "set"))
    ]
    opened = [
        SlotBindingDraft(axis=slot.axis, property_dcid=slot.property_dcid, kind="unbound")
        for slot in shape.slot_keys
        if slot.property_dcid in signature
        and (
            slot.property_dcid not in by_property
            or by_property[slot.property_dcid].kind != "unbound"
        )
    ]
    return [*kept, *opened]


def _plan_alternates(state: ResolveState) -> _AdmissionPlan:
    """Bind every alternate's places (`_bind_alternates`), apply the admission guards and
    the status rule to them (`_plan_status`), choose the readings rule 1 tries when the
    elected shape has no data (`_stand_in_indices`), apply the candidate budget
    (`_plan_admission`), and start the walk of each alternate and stand-in the plan names
    (`_start_alternate_walks`). Makes no request of its own.

    An alternate is walked at the representative it takes beside an elected shape with data,
    the case in which admission offers it; a stand-in at the one it takes beside an elected
    shape without data, the only case in which rule 1 runs."""
    state.alternate_place_roles, state.unbindable_alternates = _bind_alternates(
        state,
        [
            (i, shape)
            for i, shape in enumerate(state.capped_shapes)
            if i != state.elected_index
            and i not in state.direction_rivals
            and i not in state.breakdown_rivals
        ],
    )
    state.alternate_place_roles.update(state.direction_rivals)
    if state.elected_place_roles is not None:
        state.alternate_place_roles.update(
            dict.fromkeys(state.breakdown_rivals, state.elected_place_roles)
        )
    state.status_plan = _plan_status(state)
    state.reelection_siblings = _stand_in_indices(state)
    plan = _plan_admission(state)
    _start_alternate_walks(
        state, {i: state.capped_shapes[i] for i in plan.alt_indices}, elected_has_data=True
    )
    _start_alternate_walks(state, _sibling_shapes_of(state), elected_has_data=False)
    return plan


def _open_axes(state: ResolveState) -> OpenAxes:
    """The axes the part leaves open for the elected shape (`status_rule.open_axes`). The
    frame names a stat type when a measure word states one, a form when a reader or the
    quantity read's model read one or that model read a denominator, and a price basis when the words
    match a measurement qualifier or leave a price-basis phrase unread."""
    frame = state.frame
    measure = frame.measure if frame is not None else None
    return open_axes(
        state.shape_draft,
        stat_type_named=measure is not None and measure.stat_type.stated_by != "default",
        form_named=bool(state.quantity_requirement.named) or state.named_denominator is not None,
        price_basis_named=state.unread_price_basis
        or (frame is not None and frame.price_basis.stated_by != "default"),
    )


def _plan_status(state: ResolveState) -> StatusPlan:
    """The status rule over the elected shape's admissible alternates
    (`status_rule.plan_status`): every capped shape beside it whose roles the query's places
    bind (`state.alternate_place_roles`, which holds no `state.unbindable_alternates`) and
    that the alternate guard admits (`AlternateJudge.inadmissible`, as `_plan_admission`
    judges it), each with D read on its own assignment
    (`availability.has_data_where_asked`), whether bind nominated it, and whether the
    registry pairs it with the elected shape. The guard runs first, so `sole` means P is
    the only admissible reading with data. The measure axes the question leaves open for
    the elected shape are read here
    (`_open_axes`) and kept as `state.open_measure_axes`, which the rule and the guard both
    read. A base the question states that the quantity read
    matched to no listed denominator (`NamedDenominator.ref` `None`) is the rule's
    `base_unmatched`. A reading the model declined in favour of a refused one has no
    alternatives: the part refuses with it."""
    if state.declined is not None:
        return StatusPlan()
    state.open_measure_axes = _open_axes(state)
    judge = _guard_judge(state, elected_has_data=state.elected_has_data)
    bindable = sorted(
        (i, state.capped_shapes[i])
        for i in state.alternate_place_roles
        if i != state.elected_index
        and i not in state.direction_rivals
        and i not in state.breakdown_rivals
    )
    inadmissible = judge.inadmissible(bindable)
    bindable = [(i, shape) for i, shape in bindable if i not in inadmissible]
    twins = set(judge.twin_alternates(bindable))
    nominated = set(state.bind_result.also_plausible or ())
    series = series_at_places(state)
    menu = [
        MenuReading(
            index=i,
            shape=shape,
            has_data=has_data_where_asked(
                state, shape, series=series, assignment=state.alternate_place_roles[i]
            ),
            nominated=i in nominated,
            registry_twin=i in twins,
        )
        for i, shape in bindable
    ]
    plan = plan_status(
        state.shape_draft,
        menu,
        axes=state.open_measure_axes,
        base_unmatched=state.named_denominator is not None and state.named_denominator.ref is None,
    )
    if not state.direction_rivals and not state.breakdown_rivals:
        return plan
    # The other directions of a place named without one, and the other breakdowns of a
    # family with no total, come first beside the elected reading: the question asks which
    # it means.
    asked = (
        *(Alternative(index=i, basis="direction") for i in state.direction_rivals),
        *(Alternative(index=i, basis="breakdown") for i in state.breakdown_rivals),
    )
    return dataclasses.replace(plan, rivals=(*asked, *plan.rivals), sole=False)


def _too_many_places(state: ResolveState, exc: PlaceSetTooLargeError) -> RegionResult:
    """The answer of an elected shape that binds more places than a Spec holds. No other
    shape is elected in its place."""
    logger.info(
        "shape %r binds %d places, over the bound of %d",
        state.shape_draft.shape_id,
        exc.count,
        exc.bound,
    )
    return state.no_data("entity_set_too_large", llm_usage=state.var_usage)


def _not_admitted_places(state: ResolveState, exc: PlaceSetNotAdmittedError) -> RegionResult:
    """The answer of an elected shape a set of which has no reading its roles admit
    (`place_roles.set_choices`): `entity_role_not_admitted` naming those sets. No other
    shape is elected in its place."""
    logger.info(
        "shape %r admits no reading of %s",
        state.shape_draft.shape_id,
        [place_set.name for place_set in exc.place_sets],
    )
    return state.no_data(
        "entity_role_not_admitted",
        role_not_admitted_entities=tuple(not_admitted_entity(s) for s in exc.place_sets),
        llm_usage=state.var_usage,
    )


def _install_shape(
    state: ResolveState,
    shape: ShapeDraft,
    bindings: list[SlotBindingDraft],
    place: PlaceContext,
    unsettled: UnsettledPlace | None = None,
    *,
    index: int,
) -> None:
    """Make `shape`, the capped shape at `index`, the shape in play
    (`ResolveState.elected_index`), with its property bindings and its place context.

    Election, the settling of the elected shape's place sets and the adoption of a
    zero-observation sibling all call this, so the shape, its where binding, its roles and
    probe places, and the place warnings always describe one shape. `unsettled` holds the
    shape's places before its sets settle, for the reads that settle them again; a shape
    installed with a settled context alone has none. A model-emitted where binding without
    a property gives way to the shape's bound places (`with_where`). The place warnings of
    the shape in play sit together where it was installed: every place warning is removed
    and `place.warnings` appended, directions in place order, then the scope warning. No
    spec has been offered yet, so none of them is stamped.
    """
    state.shape_draft = shape
    state.elected_index = index
    state.bindings = with_where(bindings, place)
    state.place = place
    state.places_bound = True
    state.unsettled_place = unsettled
    install_place_warnings(state.warnings, place, departure_facts=state.departure_facts)


def _start_countries_listing(state: ResolveState, plan: _AdmissionPlan) -> None:
    """Start listing Earth's countries (`place.earth_countries`) beside the elected shape's
    walk, on a query naming no place, when a default-scope presence read may follow.

    For a subject-only recalled elected shape the listing is required work, run on the
    default executor; the elected shape's presence read awaits it (`_place_contexts`). For a
    role-bearing one whose alternates the plan admits take a default scope
    (`takes_default_scope`), it serves those alternates alone, so it runs on the alternates'
    pool (`_submit_alternates_read`) and is not started while that pool is busy; their
    presence read waits for it (`state.alternates_listing`). Either way the listing adds no
    step in series. A listing no read takes, because the walk made the shape role-bearing
    and no alternate needs it, is cancelled when the pass ends
    (`pipeline._cancel_queued_reads`): still queued, it never runs, and running, it
    finishes with its outcome discarded."""
    if not is_role_bearing(state.shape_draft):
        listing = asyncio.ensure_future(
            asyncio.to_thread(earth_countries, graph=state.graph, deadline=state.deadline)
        )
    elif any(
        takes_default_scope(state.capped_shapes[i], state.alternate_place_roles[i])
        for i in plan.alt_indices
    ):
        state.alternates_listing = _submit_alternates_read(
            earth_countries, graph=state.graph, deadline=state.deadline
        )
        if state.alternates_listing is None:
            return
        listing = asyncio.wrap_future(state.alternates_listing)
    else:
        return
    listing.add_done_callback(lambda task: task.cancelled() or task.exception())
    state.countries_listing = listing


type _Taker = tuple[ShapeDraft, Sequence[SlotBindingDraft], PlaceAssignment]
"""A shape taking a default scope, as the role-class election returned it, with the
bindings and places it is answered under."""


def _group_stat_vars(
    state: ResolveState, group: Sequence[int], takers: Sequence[_Taker] = ()
) -> list[str]:
    """The StatVars a default-scope presence read covers for `takers`, each answered under
    its own bindings (`presence_stat_vars`, as `_read_taker_stat_vars` elected it), then for
    the capped shapes at `group`'s indices,
    whose bindings are not yet known (`default_scope_stat_vars`). A taker standing for a
    held family's members past its recall reads the members a mint serves for it, since
    reading the scope of every member of a held family would cost one availability read per
    member. `takers` holds shapes as the role-class election returned them, which the capped
    menu does not."""
    return list(
        dict.fromkeys(
            [
                *(sv for taker in takers for sv in _stat_vars_of(state, taker).covered),
                *default_scope_stat_vars(
                    [
                        state.capped_shapes[i]
                        for i in dict.fromkeys(group)
                        if 0 <= i < len(state.capped_shapes)
                    ]
                ),
            ]
        )
    )


def _taker_key(taker: _Taker) -> tuple[int, tuple[str, ...], RoleKeys]:
    """The key `state.taker_stat_vars` holds `taker` under: its shape, bindings and role
    keys."""
    shape, bindings, roles = taker
    return id(shape), tuple(b.model_dump_json() for b in bindings), roles.role_keys


async def _read_taker_stat_vars(state: ResolveState, takers: Iterable[_Taker]) -> None:
    """Elect, off the event loop, the served and covered StatVars (`taker_stat_vars`) of
    each of `takers` that takes a default scope and that `state.taker_stat_vars` does not
    hold yet. For a shape standing for a held family's members the election reads the
    whole population, about 90 ms for a family of 9,641, so every reader on the event loop
    (`_group_stat_vars`, `_lacks_presence`, `_unsettled_places`) reads it from here."""
    unread = {
        _taker_key(taker): taker
        for taker in takers
        if takes_default_scope(taker[0], taker[2])
        and _taker_key(taker) not in state.taker_stat_vars
    }
    if unread:
        state.taker_stat_vars.update(
            await asyncio.to_thread(
                lambda: {
                    key: (shape, taker_stat_vars(shape, bindings, roles.role_keys))
                    for key, (shape, bindings, roles) in unread.items()
                }
            )
        )


def _stat_vars_of(state: ResolveState, taker: _Taker) -> TakerStatVars:
    """`taker`'s served and covered StatVars as `_read_taker_stat_vars` elected them. A taker
    it has not read is elected here."""
    key = _taker_key(taker)
    if key not in state.taker_stat_vars:
        shape, bindings, roles = taker
        state.taker_stat_vars[key] = (shape, taker_stat_vars(shape, bindings, roles.role_keys))
    return state.taker_stat_vars[key][1]


def _lacks_presence(
    state: ResolveState, group: Sequence[int], takers: Sequence[_Taker] = ()
) -> bool:
    """Whether `state.default_presence` lacks a StatVar the presence read covers for
    `takers` or the capped shapes at `group`'s indices (`_group_stat_vars`)."""
    known = state.default_presence
    return known is None or any(
        sv not in known.scopes for sv in _group_stat_vars(state, group, takers)
    )


async def _unsettled_places(
    state: ResolveState,
    shapes: Sequence[tuple[ShapeDraft, list[SlotBindingDraft], PlaceAssignment]],
    presence: DefaultPresence | None,
) -> list[UnsettledPlace]:
    """The places of each of `shapes` before its place sets settle, each with its bindings
    and its assignment, in order, over `presence`. A shape that takes a default scope takes
    the scopes of the StatVars a Datasets mint serves for it (`served_stat_vars`).

    Settling uses a set's reading only on roles that admit its type, so role admission is
    asked here, once per pass for each role and the type of each set's node and members
    (`UnsettledPlace.admission_questions`, `state.node_admissions`), and every question it
    has not answered is asked at once.
    """
    await _read_taker_stat_vars(state, shapes)
    places = [
        UnsettledPlace(
            shape,
            roles,
            stat_vars=_stat_vars_of(state, (shape, bindings, roles)).served
            if takes_default_scope(shape, roles)
            else (),
            presence=presence,
        )
        for shape, bindings, roles in shapes
    ]
    unasked = sorted(
        {question for place in places for question in place.admission_questions}
        - state.node_admissions.keys()
    )
    answers = await asyncio.gather(
        *[state.admission.admits(role, graph_type) for role, graph_type in unasked]
    )
    state.node_admissions.update(zip(unasked, answers, strict=True))
    admitted = frozenset(asked for asked, admits in state.node_admissions.items() if admits)
    return [dataclasses.replace(place, admitted=admitted) for place in places]


async def _place_contexts(
    state: ResolveState,
    shapes: Sequence[tuple[ShapeDraft, list[SlotBindingDraft], PlaceAssignment]],
    *,
    group: Sequence[int],
) -> list[UnsettledPlace]:
    """The places of each of `shapes` before its place sets settle, in order
    (`_unsettled_places`), over the request's presence read.

    Election and re-election call this; candidate admission reads its alternates' presence
    on their own pool (`_read_alternates_presence`). `group` holds the capped-shape indices
    of every shape whose default scope the request's presence read must cover from here
    on: at election every shape the request may probe over a default scope, at re-election
    the siblings. `shapes` are the shapes as the role-class election returned them, so the
    read covers each of them that takes a default scope whatever the capped menu holds at
    its index. When one does and `state.default_presence` still lacks one of their
    StatVars or of the group's, one presence read off-thread adds those it lacks
    (`read_default_presence`), and the result becomes `state.default_presence`. Graph
    errors propagate as retryable failures.
    """
    takers = [taker for taker in shapes if takes_default_scope(taker[0], taker[2])]
    await _read_taker_stat_vars(state, takers)
    if takers and _lacks_presence(state, group, takers):
        listing, state.countries_listing = state.countries_listing, None
        if state.default_presence is None and listing is not None:
            state.default_presence = await listing
        state.default_presence = await asyncio.to_thread(
            read_default_presence,
            graph=state.graph,
            sdmx=state.sdmx,
            stat_vars=_group_stat_vars(state, group, takers),
            known=state.default_presence,
            deadline=state.deadline,
        )
    return await _unsettled_places(state, shapes, state.default_presence)


def _lacks_default_scope(state: ResolveState, shape: ShapeDraft) -> bool:
    """Whether `shape` would take a default scope it cannot take: the query names no place
    and the shape is subject-only and index-sourced (`cannot_take_default_scope`)."""
    return not names_a_place(state) and cannot_take_default_scope(shape)


def _bind_alternates(
    state: ResolveState, alternates: Iterable[tuple[int, ShapeDraft]]
) -> tuple[dict[int, PlaceAssignment], dict[int, str]]:
    """Assign each alternate's places (`bind_alternate_place_roles`), returning assignments
    and refusal reasons by index.

    An alternate that binds its places and cannot take the default scope the query gives it
    (`_lacks_default_scope`) refuses here too and is logged. The answer does not rest on it,
    so it stays out of every presence read and out of the offer, and the request proceeds;
    the same shape elected refuses the request (`default_scope_stat_vars`).
    `_phase_place_bindings` binds every alternate once and keeps the result as
    `state.alternate_place_roles` and `state.unbindable_alternates`. A shape the role-class
    election returns with other roles is bound here again (`_walked_alternates`).
    """
    shapes = dict(alternates)
    bound, refused = bind_alternate_place_roles(state, shapes.items())
    for index in [i for i in bound if _lacks_default_scope(state, shapes[i])]:
        del bound[index]
        refused[index] = (
            "places: the query names no place and this shape is index-sourced, so it takes "
            "no default scope"
        )
        logger.info(
            "shape %r is index-sourced and takes no default scope; dropped from the offer",
            shapes[index].shape_id,
        )
    return bound, refused


def _guard_judge(state: ResolveState, *, elected_has_data: bool) -> AlternateJudge:
    """The alternate guard's judge of the elected shape under this leg's pins: its phrase,
    its bindings, the properties the engine authored, its qualifier matches, whether a
    price-basis phrase went unread, whether the elected shape has data, the leg's quantity
    requirement (`state.quantity_requirement`, which the materialise check also reads), the
    kinds of the denominators read before bind (`state.denominator_kinds`), the
    denominator the quantity read chose (`state.named_denominator`) and the measure axes
    the status rule reads as open (`state.open_measure_axes`). A judge this
    pass built for the same elected shape under equal pins is reused
    (`state.alternate_judges`)."""
    pins = query_pins(
        state.variable,
        bindings=state.bindings,
        stray_values=stray_values(state.requirements),
        engine_authored=state.measure_elected_properties | state.defaulted_properties,
        qualifier_matches=for_shape(state.query_qualifier_matches, state.shape_draft),
        qualifier_readings=state.query_qualifier_readings,
        unread_price_basis=state.unread_price_basis,
        elected_has_data=elected_has_data,
        denominator_kinds=state.denominator_kinds,
        named_denominator=state.named_denominator,
        quantity=state.quantity_requirement,
        open_axes=state.open_measure_axes,
    )
    for judge in state.alternate_judges:
        if judge.elected is state.shape_draft and judge.pins == pins:
            return judge
    judge = AlternateJudge(pins, state.shape_draft, role_keys=_place_role_keys(state))
    state.alternate_judges.append(judge)
    return judge


def _place_role_keys(state: ResolveState) -> Callable[[ShapeDraft], RoleKeys]:
    """The role keys each shape's identity binds the query's places under and the axes they
    leave open (`entity_role_keys`), by shape id: the elected shape's as admitted and each
    bound alternate's (`state.alternate_place_roles`). A shape with no binding reads none.
    The guard elects the dataset that serves each shape under them."""
    bound = [
        (state.capped_shapes[index], roles) for index, roles in state.alternate_place_roles.items()
    ]
    if state.elected_place_roles is not None:
        bound.append((state.shape_draft, state.elected_place_roles))
    keys = {shape.shape_id: roles.role_keys for shape, roles in bound}
    return lambda shape: keys.get(shape.shape_id, RoleKeys())


def _offered_shapes(state: ResolveState, shapes: Mapping[int, ShapeDraft]) -> dict[int, ShapeDraft]:
    """Each alternate or sibling narrowed to the members the guard offers it through, those
    matching the elected reading first (`AlternateJudge.offered`), so its
    representative and its walk start from a member that reads as the elected quantity."""
    judge = _guard_judge(state, elected_has_data=False)
    return {index: judge.offered(shape) for index, shape in shapes.items()}


@dataclass(frozen=True)
class _AdmissionPlan:
    """The alternates candidate admission materialises beside the shape in play, decided by
    the status rule and the admission guards alone, before any read is made for them."""

    # Why each alternate the guards reject is left out, by capped-shape index: its roles
    # cannot bind the query's places, it cannot take the default scope the query gives it,
    # or it contradicts the query's wording.
    inadmissible: dict[int, str]
    # The offered alternates that pass the guards, and those within the candidate budget.
    eligible: list[int]
    alt_indices: list[int]


def _offered(state: ResolveState) -> dict[int, CandidateBasis | None]:
    """The alternates the status rule offers beside the shape in play, by capped-shape
    index in the order admission materialises them, each with its basis (`None` for one
    that rides on a refusal).

    The registry twins come first, whether or not P has data: the equivalent-series
    diversion serves the preferred one in P's place or lists them on P. Beside an elected
    shape with data, R follows (rule 2). Beside one without, rule 1's stand-ins re-election
    found with data follow, else the nominated readings that ride on the refusal. Before the
    elected shape is read, both cases are planned, so the walks either needs start beside
    its read. The other breakdowns of a family with no total come first, whether or not P
    has data: P is one breakdown among them."""
    plan = state.status_plan
    offered: dict[int, CandidateBasis | None] = dict.fromkeys(state.breakdown_rivals, "breakdown")
    offered.update(dict.fromkeys(plan.twins, "twin"))
    read = state.mat_result is not None
    if not read or state.elected_has_data:
        offered.update((alternative.index, alternative.basis) for alternative in plan.rivals)
    if read and not state.elected_has_data and state.sibling_candidates:
        offered.update(dict.fromkeys(state.sibling_candidates, "structural"))
    elif not read or not state.elected_has_data:
        offered.update((i, offered.get(i)) for i in plan.riders)
    return offered


def _plan_admission(state: ResolveState) -> _AdmissionPlan:
    """Apply the admission guards and the candidate budget to the alternates the status
    rule offers beside the shape in play (`_offered`), making no request.

    A shape that is not a reading of the shape in play (`alternate_guard`) is rejected
    before budget or materialisation; registry twins pass for equivalent-series handling.
    A shape whose roles the query's places cannot bind, or that cannot take the default
    scope a query naming no place gives it (`state.unbindable_alternates`), is rejected the
    same way, twins included. The status rule plans over the guarded menu
    (`_plan_status`), so the guards reject here only readings a plan under other pins
    offered. The plan is made twice: before any read, and once the shape in play's data is
    known (`admit_alternates`), when a lower-ranked price basis of the same family is
    refused too on a closed price-basis axis. The budget applies once, to the eligible
    alternates in offer order, so `eligible` longer than `alt_indices` is the evidence that
    it dropped one (`CandidateSet.truncated`). Each planned alternate's basis is kept on
    the state (`ResolveState.offer_bases`).
    """
    chosen = state.elected_index
    budget = max(0, settings().max_candidates - 1)
    bindable = [(i, state.capped_shapes[i]) for i in state.alternate_place_roles]
    judge = _guard_judge(state, elected_has_data=state.elected_has_data)
    inadmissible = {**state.unbindable_alternates, **judge.inadmissible(bindable)}
    offered = _offered(state)
    eligible = [
        i
        for i in offered
        if 0 <= i < len(state.capped_shapes) and i != chosen and i not in inadmissible
    ]
    alt_indices = [] if state.force_single_shape_path else eligible[:budget]
    state.offer_bases = {i: basis for i in alt_indices if (basis := offered[i]) is not None}
    return _AdmissionPlan(inadmissible=inadmissible, eligible=eligible, alt_indices=alt_indices)


@dataclass(frozen=True)
class _WalkedAlternate:
    """An alternate as the role-class election returned it (`elect_alternate`), with the
    query's places assigned to that shape (`place_roles.bound_after_election`)."""

    alternate: _Alternate
    roles: PlaceAssignment

    @property
    def shape(self) -> ShapeDraft:
        return self.alternate.shape


_WalkKey = tuple[int, tuple[str, ...], tuple[object, ...], str, tuple[object, ...]]


def _walk_key(
    state: ResolveState, index: int, shape: ShapeDraft, rep_sv: str, roles: PlaceAssignment
) -> _WalkKey:
    """The memo key of a walk: everything `walk_alternate` reads from the request. That is
    the alternate, by capped-shape index, its members and the role keys `roles` assigns its
    places under, the representative its bindings come from, and the values the bind
    drafted on slots the elected reading lacks, which it binds where its members hold them
    (`candidates._with_stray_values`). The dataset it names is
    its shape's (`ShapeDraft.named_dataset`), or the one the part's source scope leaves
    serving under those bindings, whatever bindings the shape in play holds. The scope is
    the leg's, fixed once vetting runs."""
    role_keys = roles.role_keys
    places = (tuple(sorted(role_keys.bound)), tuple(sorted(role_keys.open)))
    strays = tuple(
        sorted((prop, tuple(sorted(values))) for prop, values in _strays_by_property(state).items())
    )
    return (index, tuple(shape.sv_arc_facts or {}), places, rep_sv, strays)


def _strays_by_property(state: ResolveState) -> dict[str, frozenset[str]]:
    """The values the bind drafted on slots the elected reading lacks
    (`requirements.stray_values`), by property."""
    by_prop: dict[str, set[str]] = {}
    for prop, value in stray_values(state.requirements):
        by_prop.setdefault(prop, set()).add(value)
    return {prop: frozenset(values) for prop, values in by_prop.items()}


def _walk_args(
    state: ResolveState, index: int, shape: ShapeDraft, rep_sv: str, roles: PlaceAssignment
) -> dict:
    return {
        "shape": shape,
        "role_keys": roles.role_keys,
        "rep_sv": rep_sv,
        "graph": state.graph,
        "deadline": state.deadline,
        "role_class_walks": state.role_class_walks,
        "scope": state.scope,
        "stray_values": _strays_by_property(state),
        "bindings": state.breakdown_rivals.get(index),
    }


def _walk_plan(
    state: ResolveState, index: int, shape: ShapeDraft, *, elected_has_data: bool
) -> tuple[str, PlaceAssignment] | None:
    """The representative the alternate at `index`, offered as `shape`, takes under
    `elected_has_data` (`alternate_representative`), and the assignment of the query's
    places it is walked under (`_walk_assignment`). `None` when it has no representative.
    The assignment is judged once per walk key of the alternate as bound
    (`state.alternate_walk_plans`), so the walk's start and its await read one judgement."""
    rep_sv = alternate_representative(
        shape,
        elected_bindings=state.bindings,
        elected_has_data=elected_has_data,
        stray_values=_strays_by_property(state),
    )
    if rep_sv is None:
        return None
    key = _walk_key(state, index, shape, rep_sv, state.alternate_place_roles[index])
    roles = state.alternate_walk_plans.get(key)
    if roles is None:
        roles = state.alternate_walk_plans[key] = _walk_assignment(state, index, shape, rep_sv)
    return rep_sv, roles


def _walk_assignment(
    state: ResolveState, index: int, shape: ShapeDraft, rep_sv: str
) -> PlaceAssignment:
    """The assignment the alternate at `index`, offered as `shape`, is walked under at
    `rep_sv`: its places with the first whole total its shape offers on a side they leave
    unnamed (`place_roles.whole_total_candidates`) whose series the alternate's own
    bindings under that total's role keys hold (`candidates.settle_alternate`,
    `place_roles.holds_whole_total`), else its places as bound
    (`state.alternate_place_roles`). The bindings are those before the walk, as binding
    repair judges the elected shape's total on its bindings before its role class is
    elected, so each alternate is walked once, whatever the shape in play binds."""
    placed = state.alternate_place_roles[index]
    for whole in whole_total_candidates(state, state.capped_shapes[index], placed):
        settled = settle_alternate(
            shape,
            whole.role_keys,
            rep_sv,
            scope=state.scope,
            stray_values=_strays_by_property(state),
            bindings=state.breakdown_rivals.get(index),
        )
        # A breakdown copy's bindings keep the elected shape's sampled claims, which select no
        # series until the walk confirms them.
        if settled is not None and holds_whole_total(
            state, settled.shape, whole, without_sampled_claims(settled.bindings)
        ):
            return whole
    return placed


def _start_alternate_walks(
    state: ResolveState, shapes: Mapping[int, ShapeDraft], *, elected_has_data: bool
) -> None:
    """Start the walk of each of `shapes`, by capped-shape index and narrowed to the members
    it is offered through (`_offered_shapes`), at the representative and under the
    assignment it is walked under (`_walk_plan`), on `_WALK_POOL`, and keep each in
    `state.alternate_walks` under its `_walk_key`. A walk already kept is not started
    again. With no walks to run (`state.role_class_walks` is `None`) the election makes no
    request, so nothing is started and `_walk` elects in place."""
    if state.role_class_walks is None:
        return
    for index, shape in _offered_shapes(state, shapes).items():
        plan = _walk_plan(state, index, shape, elected_has_data=elected_has_data)
        if plan is None:
            continue
        rep_sv, roles = plan
        key = _walk_key(state, index, shape, rep_sv, roles)
        if key in state.alternate_walks:
            continue
        state.alternate_walks[key] = _submit_walk(state, index, shape, rep_sv, roles)


def _submit_walk(
    state: ResolveState, index: int, shape: ShapeDraft, rep_sv: str, roles: PlaceAssignment
) -> Future:
    """Queue the walk of the alternate at `index`, offered as `shape`, at `rep_sv` under
    `roles` on `_WALK_POOL` (`walk_alternate`)."""
    return submit_with_context(
        _WALK_POOL, walk_alternate, **_walk_args(state, index, shape, rep_sv, roles)
    )


async def _walk(
    state: ResolveState, index: int, shape: ShapeDraft, *, elected_has_data: bool
) -> tuple[_Alternate, PlaceAssignment] | None:
    """The walk of the alternate at `index`, offered as `shape`, with the assignment of the
    query's places it answers under (`_walk_plan`), or `None` when it has no representative
    or the walk drops it. The walk is the one `_start_alternate_walks` started, awaited
    however long it waits in `_WALK_POOL`'s queue, else one queued there now. With no walks
    to run the election makes no request and runs in place."""
    plan = _walk_plan(state, index, shape, elected_has_data=elected_has_data)
    if plan is None:
        return None
    rep_sv, roles = plan
    if state.role_class_walks is None:
        alternate = walk_alternate(**_walk_args(state, index, shape, rep_sv, roles))
    else:
        key = _walk_key(state, index, shape, rep_sv, roles)
        started = state.alternate_walks.get(key)
        if started is None:
            started = state.alternate_walks[key] = _submit_walk(state, index, shape, rep_sv, roles)
        alternate = await asyncio.wrap_future(started)
    return None if alternate is None else (alternate, roles)


async def _walked_alternates(
    state: ResolveState, shapes: Mapping[int, ShapeDraft], *, elected_has_data: bool
) -> dict[int, _WalkedAlternate]:
    """Elect each alternate's role class before anything is read for its places, and bind
    the query's places to the shape the election returns. Keyed by capped-shape index, in
    `shapes` order, without the alternates the election or the binding drops. Each shape is
    walked through the members it is offered through (`_offered_shapes`), under the
    assignment of the query's places it answers under (`_walk`).

    The mint's walk can replace a role-bearing shape with a subject-only family, or the
    reverse, so the presence read, the place context and the probe places all belong to the
    shape this returns. Every alternate is walked, whether or not it declares a flow role,
    so an offered shape lists the members its own resubmit serves. The walks run beside one
    another off the event loop. A shape keeps that assignment unless the election turns it
    from role-bearing to subject-only or the reverse (`place_roles.bound_after_election`); a
    shape whose new binding refuses is dropped.
    """
    shapes = _offered_shapes(state, shapes)
    elected = await asyncio.gather(
        *[_walk(state, i, shape, elected_has_data=elected_has_data) for i, shape in shapes.items()]
    )
    walked: dict[int, _WalkedAlternate] = {}
    for (i, shape), placed in zip(shapes.items(), elected, strict=True):
        if placed is None:
            continue
        alternate, assignment = placed
        roles = bound_after_election(state, shape, assignment, alternate.shape)
        if isinstance(roles, PlaceAssignment) and not _lacks_default_scope(state, alternate.shape):
            walked[i] = _WalkedAlternate(alternate, roles)
    return walked


def _start_alternates_presence_read(
    state: ResolveState, plan: _AdmissionPlan
) -> Future[DefaultPresence] | None:
    """Start the presence read a role-bearing elected shape's alternates need, to run beside
    the elected shape's materialisation, and return it; `None` when none is needed or none
    was started.

    A role-bearing elected shape on a query naming no place makes no presence read, so a
    subject-only alternate that passed the admission guards and the candidate budget
    (`plan`) has no read to ride. Every answer that reaches candidate admission needs this
    one, so it starts here, and admission awaits it (`_read_alternates_presence`). When the
    elected result ends
    the request, the read is left to finish on its pool and nothing waits for it. With
    every worker of the pool busy it is not started, and admission tries it again.
    """
    if not any(
        takes_default_scope(state.capped_shapes[i], state.alternate_place_roles[i])
        for i in plan.alt_indices
    ):
        return None
    return _submit_alternates_presence_read(state, _group_stat_vars(state, plan.alt_indices))


def _submit_alternates_read[T](read: Callable[..., T], /, **kwargs: object) -> Future[T] | None:
    """Submit `read(**kwargs)` to `_ALTERNATES_POOL` and return it, or return `None` when
    every worker of the pool is busy. A read that stalls holds one of the pool's workers
    until its source answers, and none of the default executor's, which required work runs
    on."""
    if not _ALTERNATES_SLOTS.acquire(blocking=False):
        return None
    try:
        future = submit_with_context(_ALTERNATES_POOL, read, **kwargs)
    except BaseException:
        _ALTERNATES_SLOTS.release()
        raise
    future.add_done_callback(lambda _: _ALTERNATES_SLOTS.release())
    return future


def _presence_after_listing(
    *,
    listing: Future[DefaultPresence] | None,
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    stat_vars: list[str],
    known: DefaultPresence | None,
    deadline: float | None,
) -> DefaultPresence:
    """`read_default_presence`, taking Earth's countries from `listing` when the read knows
    no presence yet. The listing runs on another worker of the alternates' pool, so waiting
    for it holds no worker it needs, and its failure is the read's."""
    if known is None and listing is not None:
        known = listing.result()
    return read_default_presence(
        graph=graph, sdmx=sdmx, stat_vars=stat_vars, known=known, deadline=deadline
    )


def _submit_alternates_presence_read(
    state: ResolveState, stat_vars: list[str]
) -> Future[DefaultPresence] | None:
    """Submit the presence read of `stat_vars` (`_presence_after_listing`) on
    `_ALTERNATES_POOL` (`_submit_alternates_read`), or return `None` when none is free. The
    listing of Earth's countries started on that pool beside the elected shape's walk
    (`state.alternates_listing`) gives its countries."""
    return _submit_alternates_read(
        _presence_after_listing,
        listing=state.alternates_listing,
        graph=state.graph,
        sdmx=state.sdmx,
        stat_vars=stat_vars,
        known=state.default_presence,
        deadline=state.deadline,
    )


def merge_presence(state: ResolveState, presence: DefaultPresence | None) -> None:
    """Merge `presence`, a read made over the request's presence when it started, into
    `state.default_presence`, so a read made since keeps what it added and the pass's
    unsettled scopes (`pipeline._scope_unsettled`) include those `presence` left."""
    if presence is None:
        return
    known = state.default_presence
    state.default_presence = presence if known is None else known.merged(presence)


def record_unevaluated(
    state: ResolveState, unread: Mapping[int, AlternateUnevaluatedReason]
) -> None:
    """Record the capped-shape indices in `unread` as left unevaluated by a presence read that
    did not complete, each with the reason, as `_read_alternates_presence` records its own:
    the pass states them (`alternates_unevaluated`), tries no other optional presence read,
    and its result stays out of the response cache."""
    if not unread:
        return
    if state.alternates_presence_failure is None:
        state.alternates_presence_failure = next(iter(unread.values()))
    state.unevaluated_alternates.update(unread)
    logger.warning(
        "a presence read after unbinding did not complete; shapes %r are left unevaluated; "
        "trace_id=%s",
        [state.capped_shapes[i].shape_id for i in unread],
        get_trace_id(),
    )


async def _await_alternates_presence(
    state: ResolveState, read: Future[DefaultPresence]
) -> AlternateUnevaluatedReason | None:
    """Await `read` and merge its presence into the request's, or return why it did not
    complete. The read was made over the presence known when it was submitted, so a read
    made since (a re-election's) keeps what it added. The read serves alternates only, so its
    failure leaves them unevaluated and the elected answer stands."""
    try:
        merge_presence(state, await asyncio.wrap_future(read))
    except (TimeoutError, dc_graph.GraphDeadlineExceededError):
        return "timed_out"
    except (GraphInfraError, dc_graph.GraphReadError, dc_graph.GraphCapExceededError):
        return "upstream_error"
    return None


async def _read_alternates_presence(
    state: ResolveState,
    shapes: Mapping[int, _Taker],
    *,
    group: Sequence[int],
) -> dict[int, AlternateUnevaluatedReason]:
    """Read the presence the alternates in `shapes` need for their default scopes, and
    return the indices left without it, in order, each with the reason.

    `shapes` holds each alternate by capped-shape index as the role-class election returned
    it, with its bindings and the places bound to that shape (`_walked_alternates`), so the
    read covers the StatVars the alternate is probed through (`_group_stat_vars`). The read started beside a role-bearing
    elected shape's materialisation (`state.alternates_presence`) is awaited first. Presence
    the request still lacks is read over those shapes and the rest of `group`, the shapes a
    widened search may add, on `_ALTERNATES_POOL`
    (`_submit_alternates_presence_read`), and the answer waits for it as for a read in
    series.

    The read is optional: it serves alternates only. One rule covers every way it can fail
    to complete. With every worker of the pool busy it is not made (`no_capacity`); a
    presence source that fails gives `upstream_error`, and one that times out `timed_out`.
    The failure is recorded on the pass (`state.alternates_presence_failure`), and the
    pass tries no other optional presence read after it, whichever alternates it would have
    served; the UN System pass starts from a fresh state and reads its own upstream, so it
    reads independently. Each alternate whose default scope the read would give is returned and
    recorded in `state.unevaluated_alternates`; the pass's result states them
    (`alternates_unevaluated`) and stays out of the response cache.
    """
    takers = {i: taker for i, taker in shapes.items() if takes_default_scope(taker[0], taker[2])}
    if not takers:
        return {}
    await _read_taker_stat_vars(state, takers.values())
    others = [i for i in group if i not in shapes]
    reason = state.alternates_presence_failure
    started, state.alternates_presence = state.alternates_presence, None
    if reason is None and started is not None:
        reason = await _await_alternates_presence(state, started)
    if reason is None:
        if not _lacks_presence(state, others, list(takers.values())):
            return {}
        read = _submit_alternates_presence_read(
            state, _group_stat_vars(state, others, list(takers.values()))
        )
        if read is None:
            reason = "no_capacity"
        else:
            reason = await _await_alternates_presence(state, read)
            if reason is None:
                return {}
    state.alternates_presence_failure = reason
    unread = dict.fromkeys(
        (i for i, taker in takers.items() if _lacks_presence(state, [], [taker])), reason
    )
    if unread:
        state.unevaluated_alternates.update(unread)
        logger.warning(
            "the alternates presence read did not complete (%s); shapes %r are left "
            "unevaluated; trace_id=%s",
            reason,
            [state.capped_shapes[i].shape_id for i in unread],
            get_trace_id(),
        )
    return unread


async def _phase_facet_requests(state: ResolveState) -> RegionResult | None:
    """Bind parsed-door unit text or return its terminal refusal.

    Resubmit calls have no unit text. Expressible unit requests continue with narrowed
    facets. Parse errors and unsupported units terminate here.
    """
    # Unit parse failures return ``unit_not_resolved`` or ``unit_request_unsupported``. The
    # binder emits only ISO codes and role shapes, every one expressible, and a bound unit
    # narrows later reads. Completion reads the conversion tables' base year only
    # for a constant-price phrase that names no year, off the event loop because the first
    # such read in a process goes to the graph.
    if state.unit_text is not None:
        graph, deadline = state.graph, state.deadline
        policy = UnitDefaultPolicy(
            currency=settings().unit_default_currency,
            base_year=lambda: declared_base_year(graph, deadline=deadline),
        )
        unit_outcome = await asyncio.to_thread(
            bind_unit_text,
            state.unit_text,
            policy=policy,
            bound_roles=frozenset(role for _, role in state.place.roles.roles),
        )
        if unit_outcome.code == "unit_not_resolved":
            return state.no_data(
                "unit_not_resolved",
                llm_usage=state.var_usage,
                unit_alternatives=tuple(
                    UnitAlternative(currency=a.currency, label=a.label)
                    for a in unit_outcome.alternatives[:SEARCH_MAX_UNIT_ALTERNATIVES]
                ),
            )
        if unit_outcome.code == "unit_request_unsupported":
            return _unsupported(
                state,
                unit_outcome.code,
                pointer="/identity/facets/unit",
                detail=unit_outcome.message,
                llm_usage=state.var_usage,
            )
        _unit_request = state.unit_request = IdentityUnitRequest(
            currency=unit_outcome.currency,
            prices=unit_outcome.prices,
            base_year=unit_outcome.base_year,
        )
        _bound_unit_outcome = state.bound_unit_outcome = unit_outcome
        # Use the literal predicate for display facets and coverage. Observation fetches use
        # the request-aware election so converted native facets remain eligible.
        _unit_facet_filter = state.unit_facet_filter = {
            "unit": FacetAxisFilter(
                predicate=lambda f: (
                    f.unit is not None
                    and _unit_satisfies(state.graph, _unit_request, f.unit, deadline=state.deadline)
                )
            )
        }

    # Read the optional outcome from state after the unit-text branch.
    _bound_unit_outcome = state.bound_unit_outcome

    # Share bound axes and request basis across candidate and single-shape observation fetches.
    _bound_facet_axes: dict[str, FacetBasis] = (
        {"unit": "bound"} if _bound_unit_outcome is not None else {}
    )
    state.bound_facet_axes = _bound_facet_axes
    _unit_request_basis = state.unit_request_basis = (
        _bound_unit_outcome.request_basis if _bound_unit_outcome is not None else None
    )

    return None


type _MaterialiseOutcome = Materialised | NoDataDraft | MaterialisedCandidates | UnsupportedDraft


def _materialise_task(
    state: ResolveState,
) -> asyncio.Task[tuple[PlaceContext, _MaterialiseOutcome]]:
    """Start materialising the elected shape over `state.bindings` off-thread, and return
    the place context it is read in with its result.

    A shape whose read decides a place set settles its sets on this read
    (`place_context.settle_sets`): it is materialised at a set's node and, where the node
    holds no data in the years asked for, at the set's members, from one facet read. Any
    other shape is materialised in the context it was installed with.
    """
    shape, bindings, presence = state.shape_draft, list(state.bindings), state.default_presence
    unsettled = state.unsettled_place
    target = unsettled if unsettled is not None and not unsettled.decided else state.place

    def read(place: PlaceContext, facet_batch: FacetBatch | None) -> _MaterialiseOutcome:
        return run_materialise(
            shape,
            with_where(bindings, place),
            place.roles.resolved_entities,
            graph=state.graph,
            sdmx=state.sdmx,
            date_request=state.date_request,
            deadline=state.deadline,
            place_dcids=place.probe_places,
            default_presence=place.presence_for(presence),
            open_axes=place.roles.open_axes,
            coverage=state.coverage,
            facet_batch=facet_batch,
        )

    def settle() -> tuple[PlaceContext, _MaterialiseOutcome]:
        return settle_sets(target, read)

    # Match the observation fetch's ``needs_dates=True`` cache key when a preview is enabled,
    # including dateless queries. The context scope applies only to this task.
    with force_full_facets(state.include_observations or state.stream_observations):
        return asyncio.create_task(asyncio.to_thread(settle))


class _RelaxedPartRefused(Exception):
    """No table serves the part under the bindings an unbinding leaves; `result` is its
    refusal (`phase_election._settle_served_dataset`)."""

    def __init__(self, result: RegionResult) -> None:
        super().__init__(result.no_data_reason)
        self.result = result


async def _unbind(
    state: ResolveState, unrealisable: BoundValuesUnrealisable, role_keys: RoleKeys
) -> RegionResult | None:
    """Unbind values of `unrealisable` (`unbind_unrealisable`) and settle the dataset the
    part is served from again under what remains, returning the refusal of a part no table
    then serves. The dataset is named from the bindings wherever the index decides the
    part's membership (`bound_values.part_source`), and a query the index did not decide
    can be one it decides once a value on a property no index row declares is unbound.
    On a part a table serves, the query records no rating reads are read again over the
    bindings so settled (`_resettle_unbound_part`), as binding repair keeps them: "female"
    in "female aged X" reads no bound code once Sex=Female is unbound."""
    unbind_unrealisable(state, unrealisable)
    return await _resettle_unbound_part(state, role_keys)


def _judge_sampled_claims(state: ResolveState, role_keys: RoleKeys) -> frozenset[str]:
    """The properties of the claims a recall sample alone backs
    (`phase_election.sampled_claims`) that the family the walk settled refutes, once each
    claim it confirms is no longer marked sampled.

    The walk read each claimed slot open (`phase_election.without_sampled_claims`), and its
    family (`ShapeDraft.sv_arc_facts` of the shape it settled) confirms a claim when every
    member asserts the claimed value (`sole_shared_value`), the rule a claim on a complete
    population meets at settle. A confirmed claim keys the walk (`ResolveState.walked`) to
    the query binding its value (`WalkedMembers.binding`): every walked member asserts it,
    so the walk of that query settles the same members, and the identity minted under the
    bindings reads them (`stated_members`). With no family settled (`ResolveState.walked`
    is None), every claim stands on its sample."""
    claims = sampled_claims(state.bindings)
    if not claims or state.walked is None:
        return frozenset()
    constraints = [
        read_constraints(arcs) for arcs in (state.shape_draft.sv_arc_facts or {}).values()
    ]
    confirmed = {
        prop for prop, value in claims.items() if sole_shared_value(constraints, prop) == value
    }
    refuted = frozenset(claims.keys() - confirmed)
    if not confirmed:
        return refuted
    state.bindings = [
        b.model_copy(update={"sampled": False}) if b.property_dcid in confirmed else b
        for b in state.bindings
    ]
    bound_query = shape_query(
        state.shape_draft,
        without_values(state.bindings, refuted),
        role_keys.bound,
        open_role_keys=role_keys.open,
    )
    for prop in confirmed:
        state.walked = state.walked.binding(prop, bound_query.slots[prop])
    return refuted


async def _unbind_refuted_claims(
    state: ResolveState, refuted: frozenset[str], role_keys: RoleKeys
) -> RegionResult | None:
    """Unbind each slot of `refuted`, claims a recall sample backed and the walked family
    refutes (`_judge_sampled_claims`), and settle the part again as an unbinding does
    (`_resettle_unbound_part`). The request's words never named the value, so the answer
    records no requirement and names no departure for it."""
    state.bindings = without_values(state.bindings, refuted)
    state.forget_bound(refuted)
    return await _resettle_unbound_part(state, role_keys)


async def _resettle_unbound_part(state: ResolveState, role_keys: RoleKeys) -> RegionResult | None:
    """Settle the dataset the part is served from under the bindings an unbinding leaves,
    returning the refusal of a part no table then serves. On a part a table serves, the
    query records no rating reads are read again over the bindings so settled
    (`phase_rank_bind._query_records`). Coverage is judged once, on what the read then
    serves (`phase_election.judge_coverage`), or before the refusal of a part no table
    serves."""
    if (refused := await _settle_served_dataset(state, role_keys)) is not None:
        return refused
    record_reading(state, _query_records(state))
    return None


async def _run_materialise_task(state: ResolveState) -> _MaterialiseOutcome:
    """Run materialisation off-thread and time it, and install the place context the
    elected shape's place sets settle to (`_materialise_task`). Bound values that together
    match no member lose a minimal set and the shape is resolved again
    (`unrealisable.unbind_unrealisable`) until its bindings can be read, over a presence
    read that covers what the resolve under the new bindings reads (`_cover_unbound`).

    Raises `PlaceSetTooLargeError` when the places the sets settle to exceed the bound on
    the places a Spec binds, and `_RelaxedPartRefused` when no table serves the part under
    the bindings an unbinding leaves (`_unbind`)."""
    t0 = now_ms()
    materialise_task = _materialise_task(state)
    try:
        place, mat_result = await materialise_task
        while isinstance(mat_result, BoundValuesUnrealisable):
            refused = await _unbind(
                state,
                mat_result,
                entity_role_keys(
                    state.place.roles.resolved_entities, open_axes=state.place.roles.open_axes
                ),
            )
            if refused is not None:
                raise _RelaxedPartRefused(refused)
            await _cover_unbound(state)
            materialise_task = _materialise_task(state)
            place, mat_result = await materialise_task
        if place != state.place:
            _install_shape(
                state,
                state.shape_draft,
                state.bindings,
                place,
                state.unsettled_place,
                index=state.elected_index,
            )
        state.mat_result = mat_result
    finally:
        # Cancel and await unfinished materialisation so cancellation cannot detach it.
        if not materialise_task.done():
            materialise_task.cancel()
            await asyncio.gather(materialise_task, return_exceptions=True)
    state.timing["materialise"] = now_ms() - t0
    await _emit_step(
        state.emit,
        state.variable_index,
        "materialise",
        ms=state.timing["materialise"],
    )
    return mat_result


async def _cover_unbound(state: ResolveState) -> None:
    """Top `state.default_presence` up, off-thread, with the scopes the elected shape's
    resolve under `state.bindings` reads and the read made under its earlier bindings lacks
    (`presence_covering`). Unbinding can widen a held shape's served members past the ones
    that read covered. Nothing is read off a default scope."""
    presence = state.default_presence
    if presence is None or state.place.default_scope is None:
        return
    role_keys = entity_role_keys(
        state.place.roles.resolved_entities, open_axes=state.place.roles.open_axes
    )
    state.default_presence = await asyncio.to_thread(
        presence_covering,
        presence,
        state.shape_draft,
        state.bindings,
        role_keys,
        graph=state.graph,
        sdmx=state.sdmx,
        deadline=state.deadline,
    )


def _members_holding_bound_values(
    shape: ShapeDraft, bound_values: Mapping[str, list[str]]
) -> dict[str, dict]:
    """Return the members `shape` stands for (`shape_population_arcs`) that assert every
    property `bound_values` binds, at a bound value, with their arcs.

    A qualifier variant measures the elected quantity only when it keeps the elected
    bindings: nominal GDP beside inflation-adjusted GDP both hold
    ``activitySource=GrossDomesticProduction``, and GNI at purchasing power parity holds
    another value. No bound value admits every member.
    """
    return {
        sv: arcs
        for sv, arcs in shape_population_arcs(shape).items()
        if all(read_constraints(arcs).get(prop) in values for prop, values in bound_values.items())
    }


def _stand_in_members(shape: ShapeDraft, bindings: Sequence[SlotBindingDraft]) -> dict[str, dict]:
    """The members of the stand-in `shape` that keep the elected `bindings`, with their arcs:
    those asserting every property a binding the request named binds at a bound value
    (`_members_holding_bound_values`), and no value other than the one a value the request
    did not name binds (`sole_option_properties`). Such a value is the elected family's
    own description of itself, so a member stating nothing on its property reads the same
    quantity another way: the IDS gross national income states its income as
    ``economicActivityType`` beside the Atlas method's ``activitySource=GrossNationalIncome``."""
    unnamed = sole_option_properties(bindings)
    asked = bound_values_from_bindings([b for b in bindings if b.property_dcid not in unnamed])
    sole = bound_values_from_bindings([b for b in bindings if b.property_dcid in unnamed])
    return {
        sv: arcs
        for sv, arcs in _members_holding_bound_values(shape, asked).items()
        if all(read_constraints(arcs).get(prop) in (None, *values) for prop, values in sole.items())
    }


def _stand_in_indices(state: ResolveState) -> list[int]:
    """The capped-shape indices of the readings status rule 1 tries in the elected shape's
    place when it has no observations at its places, in the order the rule takes them: its
    equivalent readings, then its measurement-qualifier siblings that keep the elected
    bindings (`_stand_in_members`). Each passes the alternate guard
    (`AlternateJudge.inadmissible`) judged beside an elected shape without data, so no
    default price basis is imposed, within the candidate budget. A part with a registry twin
    of the elected shape takes none: the equivalent-series diversion answers with the twin
    (`candidate_collapse._apply_twin_diversion`). They depend on the elected shape and the
    bind step alone, so they are known before any read."""
    plan = state.status_plan
    if plan.twins:
        return []
    indices = [
        *plan.equivalents,
        *(i for i in plan.stand_ins if _stand_in_members(state.capped_shapes[i], state.bindings)),
    ]
    if indices:
        inadmissible = _guard_judge(state, elected_has_data=False).inadmissible(
            [(i, state.capped_shapes[i]) for i in indices]
        )
        indices = [i for i in indices if i not in inadmissible]
    return indices[: max(0, settings().max_candidates - 1)]


def _sibling_shapes_of(state: ResolveState) -> dict[int, ShapeDraft]:
    """Each re-election sibling (`state.reelection_siblings`) by capped-shape index, holding
    the members it is probed and answered through: those that keep the elected bindings
    among every member it stands for (`_stand_in_members`), and those alone
    (`narrowed_to`)."""
    return {
        i: narrowed_to(
            state.capped_shapes[i], _stand_in_members(state.capped_shapes[i], state.bindings)
        )
        for i in state.reelection_siblings
    }


async def _reelect_zero_observation_sibling(
    state: ResolveState, mat_result: Materialised
) -> Materialised:
    """Status rule 1's stand-ins (`state.reelection_siblings`) when the elected shape has no
    observations at its places (``has_data=False``, ``available_at_place=False``): its
    equivalent readings, then its measurement-qualifier siblings.

    Each stand-in's members are settled first by the mint's walk (`_walked_alternates`), so
    one that answers lists the members its own resubmit serves, and
    ``_materialise_alt_shapes`` then prewarms facets and resolves them. Each settles its own
    place sets on its own read, for the shape the election returned (`_place_contexts`,
    `place_context.settle_sets`), or takes the default scopes of its own StatVars when it is
    subject-only and the query names no place. The elected shape's presence read covers
    the scopes of the stand-ins' recalled members (`_phase_place_bindings`), so a read is
    made here only for a stand-in whose walk replaced its members, or for the members a mint
    serves (`_group_stat_vars`) of one narrowed from a held family's members past its
    recalled ones. Stand-ins with separate probe places are read concurrently, so their
    facet reads take one serial step between them.

    The first equivalent reading with data answers. Otherwise a qualifier sibling whose walk
    elects a shape of the elected shape's role class (`status_rule.shares_role_class`)
    stands in: one with data answers, and several are offered as candidates without the
    elected shape, kept as `state.sibling_candidates`, which candidate admission takes in
    place of materialising them (`admit_alternates`). An answering stand-in is installed in
    place of the elected shape with its bindings and place context (`_install_shape`),
    replaces the elected result, and warns `elected_shape_no_data` after its place warnings.
    Its `elected` starts as the stand-in's draft label and becomes the shape label of the
    Spec built for the stand-in, on the definite path and among candidates alike
    (`ResolveState.state_reelected_reading`). A part refused before any such Spec is built
    keeps the draft label.
    A winner whose members the walk left unsettled states it once its result is read
    (`interpret_materialised`). Blocked or empty probes leave the result and the place
    context unchanged.
    """
    _sibling_shapes = _sibling_shapes_of(state)
    # A sibling's members are settled before its places, presence and context, which belong
    # to the shape it is answered through.
    _walked = await _walked_alternates(state, _sibling_shapes, elected_has_data=False)
    # A qualifier sibling stands in only as a reading of the elected shape's role class, as
    # its walk elects it; an equivalent reading of the five-tuple stands in whatever its
    # role class.
    equivalents = frozenset(state.status_plan.equivalents)
    _walked = {
        i: walked
        for i, walked in _walked.items()
        if i in equivalents or shares_role_class(state.shape_draft, walked.shape)
    }
    places = await _place_contexts(
        state,
        [(walked.shape, walked.alternate.bindings, walked.roles) for walked in _walked.values()],
        group=[],
    )
    # Siblings sharing probe places share one materialise call, so its facet prewarm covers
    # them in one read, and the calls of separate probe places run side by side.
    by_scope: dict[tuple[str, ...] | None, list[tuple[_Alternate, UnsettledPlace]]] = {}
    for walked, place in zip(_walked.values(), places, strict=True):
        try:
            scope = place.unread().probe_dcids
        except PlaceSetRefusedError:
            # A sibling that cannot bind its places is not probed or offered.
            continue
        by_scope.setdefault(scope, []).append((walked.alternate, place))
    _scope_results: list[_AlternatesRead] = await asyncio.gather(
        *[
            asyncio.to_thread(
                _materialise_alt_shapes,
                [alternate for alternate, _ in members],
                places=[place for _, place in members],
                presence=state.default_presence,
                named_place_dcids=state.standard_place_dcids,
                graph=state.graph,
                sdmx=state.sdmx,
                date_request=state.date_request,
                deadline=state.deadline,
                coverage=state.coverage,
            )
            for members in by_scope.values()
        ]
    )
    for _scope_read in _scope_results:
        merge_presence(state, _scope_read.presence)
        record_unevaluated(
            state,
            {
                i: reason
                for i in _walked
                for alternate, reason in _scope_read.unevaluated
                if _walked[i].alternate is alternate
            },
        )
    # The role-class election can replace a sibling's shape, so its group key, which the
    # election leaves unchanged, maps a result back to its menu index.
    _index_by_group = {_shape_group_key(_walked[i].shape): i for i in _walked}
    # An over-cap sibling, or one whose presence read after unbinding did not complete, is
    # unevaluated. Keep the elected no-observations result rather than treating that sibling
    # as disproven or surfacing a cap refusal.
    if any(
        _scope_read.blocking is not None or _scope_read.unevaluated
        for _scope_read in _scope_results
    ):
        return mat_result
    _sibling_candidates = [
        cand for _scope_read in _scope_results for cand in _scope_read.candidates
    ]
    if not _sibling_candidates:
        return mat_result
    # ``state.reelection_siblings`` ascend, so this restores menu order across scope groups.
    _sibling_candidates.sort(key=lambda cand: _index_by_group[_shape_group_key(cand.shape)])

    _answering = [
        cand
        for cand in _sibling_candidates
        if _index_by_group[_shape_group_key(cand.shape)] in equivalents
    ]
    if not _answering and len(_sibling_candidates) == 1:
        _answering = _sibling_candidates
    if _answering:
        _winner = _answering[0]
        _no_data_label = state.shape_draft.label
        _install_shape(
            state,
            _winner.shape,
            _winner.bindings,
            _winner.place,
            index=_index_by_group[_shape_group_key(_winner.shape)],
        )
        state.walked = _winner.walked
        state.mat_result = _winner.mat
        mat_result = _winner.mat
        state.reelection_warning = _elected_sibling_warning(
            no_data=[_no_data_label], elected=[_winner.shape.label]
        )
        state.warnings.append(state.reelection_warning)
    else:
        state.sibling_candidates = {
            _index_by_group[_shape_group_key(cand.shape)]: cand for cand in _sibling_candidates
        }
    return mat_result


async def _phase_materialise(state: ResolveState) -> RegionResult | None:
    """Materialise the elected shape, settling its place sets on that read, try status
    rule 1's stand-ins when it has no observations at its places
    (`_reelect_zero_observation_sibling`), and apply per-capita denominator precedence. A part whose
    model declined every admissible reading refuses with the one it named instead, before
    any read (`_declined_refusal`)."""
    if state.declined is not None:
        return await _declined_refusal(state)
    try:
        mat_result = await _run_materialise_task(state)
    except PlaceSetTooLargeError as exc:
        return _too_many_places(state, exc)
    except PlaceSetNotAdmittedError as exc:
        return _not_admitted_places(state, exc)
    except _RelaxedPartRefused as exc:
        return exc.result

    if (
        state.reelection_siblings
        and isinstance(mat_result, Materialised)
        and not mat_result.has_data
        and mat_result.available_at_place is False
    ):
        mat_result = await _reelect_zero_observation_sibling(state, mat_result)

    # Classify terminal outcomes and append the answering result's warnings, after any
    # re-election, so they describe the shape in ``state.shape_draft``.
    state.opposite_labels = memoised_labels(state.label_dcids)
    outcome = interpret_materialised(
        mat_result,
        boundary_roles=state.place.roles.roles,
        label_of=state.opposite_labels,
        warnings=state.warnings,
        unnamed_wholes=state.place.roles.unnamed_wholes,
    )
    # A discovered scope exceeding its year limit is unsupported. Handle it before data checks.
    if isinstance(outcome, UnsupportedMaterialise):
        return _unsupported(
            state,
            outcome.code,
            pointer=outcome.pointer,
            detail=outcome.detail,
            llm_usage=state.var_usage,
        )

    # Role-index declaration mismatches return bare no-data. Attach the elected identity
    # through the shared role-incompatible result to preserve it downstream.
    if isinstance(outcome, NoDataMaterialise) and outcome.reason == "role_incompatible":
        return await _role_incompatible_no_data(state)

    # Read the answer over the context its result settles: every later probe and the preview
    # bind a discovered scope's members, on the axis the scope reports, including origin-only
    # scopes, or the default scope of the StatVars the answer serves.
    # The open axes narrow to those the elected members serve, and the shape declares their
    # roles.
    if isinstance(mat_result, Materialised):
        state.place = settle_place(state.place, mat_result, state.warnings)
        state.shape_draft = settled_shape(state.shape_draft, mat_result)

    # Preserve no-data precedence. ``no_matching_slice`` and ``variable_not_resolved`` win.
    # Otherwise an elected shape that does not serve the leg's quantity
    # (`quantity.serves_quantity`: an amount, or a share of GNI, for "aid per capita"; a
    # dollar level for "debt to GDP ratio"; a share of GNI for "debt to GDP") outranks
    # ``no_observations`` or base-measure data.
    if not serves_leg_quantity(state, state.shape_draft):
        # Defer a place-less missing slice (no member asserts every bound value, so no
        # availability verdict exists) so the pipeline can mint ``no_matching_slice`` with
        # its flagged match. A bound member without data is no missing slice.
        _slice_mint_pending = (
            state.destination_dcid is None
            and isinstance(mat_result, Materialised)
            and not mat_result.has_data
            and mat_result.available_at_place is None
        )
        _elected = _elect_no_data_reason(
            mat_result.reason if isinstance(mat_result, NoDataDraft) else None,
            "no_matching_slice" if _slice_mint_pending else None,
            fallback="denominator_not_available",
        )
        if _elected == "denominator_not_available" and any(
            serves_leg_quantity(state, state.capped_shapes[i])
            for i in _plan_admission(state).alt_indices
        ):
            # An alternate the fan-out admits serves the quantity, so the fan-out runs, and
            # the refusal waits for an outcome in which no reading that serves it survives
            # (`refuse_unserved_quantity`).
            state.quantity_refusal = mat_result
            return None
        if _slice_mint_pending and _elected == "no_matching_slice":
            # The single-shape path attaches the flagged match after Spec assembly.
            state.force_single_shape_path = True
        elif _elected == "denominator_not_available":
            return await _denominator_refusal(state, mat_result)
        else:
            return state.no_data(_elected, llm_usage=state.var_usage)

    return None


def serves_leg_quantity(state: ResolveState, shape: ShapeDraft) -> bool:
    """Whether `shape` serves the leg's quantity: every form it names and the denominator
    the quantity read chose (`quantity.serves_quantity`)."""
    return serves_quantity(
        state.quantity_requirement, state.named_denominator, shape, state.denominator_kinds
    )


async def refuse_unserved_quantity(state: ResolveState) -> RegionResult | None:
    """The `denominator_not_available` refusal the materialise gate deferred to the
    fan-out (`state.quantity_refusal`), once no reading that serves the quantity survived
    it (`serves_leg_quantity`). `None` when the gate deferred nothing."""
    if state.quantity_refusal is None:
        return None
    return await _denominator_refusal(state, state.quantity_refusal)


def note_denominator_unverified(
    state: ResolveState, shape: ShapeDraft, *, spec_id: str | None
) -> None:
    """Warn `denominator_unverified` when `shape`, a shape the answer serves, is not
    checked against the denominator the question names (`quantity.check_denominator`):
    it states its base only in its name, or it divides by a base the quantity read
    matched to none of the listed ones. The warning carries the `spec_id` of the spec
    `shape` answers through, definite or candidate."""
    named = state.named_denominator
    if named is not None and check_denominator(shape, named) == "unverified":
        arc = stated_denominator(shape)
        state.warnings.append(
            denominator_unverified_warning(
                named,
                divides=arc is not None,
                arc_name=_denominator_name(state, arc) if arc is not None else None,
                spec_id=spec_id,
            )
        )


# The words for the base a schema-term denominator names (`quantity.schema_denominator_kind`).
_SCHEMA_BASE_WORDS = {"person": "people", "area": "land area"}


def _denominator_name(state: ResolveState, arc: str) -> str | None:
    """The name of the base `arc` divides by: words for a schema term, else the graph's
    name for the StatVar (`state.denominator_reads`), `None` when the graph gives none."""
    if (kind := schema_denominator_kind(arc)) is not None:
        return _SCHEMA_BASE_WORDS.get(kind)
    read = state.denominator_reads.get(arc)
    return read.name if read is not None else None


def denominator_unverified_warning(
    named: NamedDenominator, *, divides: bool, arc_name: str | None, spec_id: str | None
) -> Warning:
    """Describe an answer the search could not check against the denominator the question
    names, `named`, quoting the question's own words for it. A series that `divides` by a
    stated base, `arc_name` when known, names that base beside them; any other series
    states no base, so only its name says whether it divides by the named one."""
    if divides:
        found = f'"{arc_name}"' if arc_name else "a base the graph does not name"
        message = (
            f'the question names "{named.base_words}" as the base; this series divides '
            f"by {found}, which the search could not match to it."
        )
    else:
        message = (
            f"this series states no base the search could check, so only its name says "
            f'whether it divides by "{named.base_words}".'
        )
    return Warning(code=DENOMINATOR_UNVERIFIED, severity="warn", message=message, spec_id=spec_id)


async def _declined_refusal(state: ResolveState) -> RegionResult:
    """Refuse with the reading the bind model named as nearest when it declined every
    admissible one (`ResolveState.declined`), standing in as the elected shape with its
    bindings and places settled. A reading that fails the form the question names refuses
    ``denominator_not_available`` (`_denominator_refusal`), and one that cannot bind a code
    the question names refuses ``qualifier_not_bound``, naming the terms its flagged
    match's trace leaves unmet (`requirements.settle_answer`), those codes among them
    (election vetting records them). Either flags the reading, with its level reading as
    nearest_real, and states as open the axes the frame asks every member of that its
    members declare a role on (`_declined_open_roles`), as its definite answer would."""
    declined = state.declined
    assert declined is not None
    members, open_roles = _unread_answer_members(state)
    if declined.pinned_failure != "pinned_code":
        return await _denominator_refusal(state, None, members=members, open_roles=open_roles)
    flagged = await _elected_flagged_match(state, members=members, open_roles=open_roles)
    return state.no_data(
        "qualifier_not_bound",
        flagged_match=(flagged,),
        nearest_real=await _nearest_real_for(state, level=_level_relaxation(state)),
        unbound_qualifiers=settle_answer(state, state.shape_draft, state.bindings),
        llm_usage=state.var_usage,
    )


async def refuse_unread_answer(state: ResolveState) -> RegionResult:
    """Refuse ``qualifier_not_bound`` on the elected answer, `state.shape_draft` read under
    `state.bindings`, when a read raised a refusal of the whole request before the answer
    was served (`pipeline._run_phases`). The elected answer leaves terms unmet, so the
    part answers nothing on that series and the series refuses no request. The refusal
    stands in for definite assembly's (`phase_answer._assemble_carve_outs`): it flags the
    elected answer with its bindings and places settled, as `_declined_refusal` does, and
    names the terms its trace leaves unmet (`requirements.settle_answer`)."""
    members, open_roles = _unread_answer_members(state)
    flagged = await _elected_flagged_match(state, members=members, open_roles=open_roles)
    return state.no_data(
        "qualifier_not_bound",
        flagged_match=(flagged,),
        unbound_qualifiers=settle_answer(state, state.shape_draft, state.bindings),
        llm_usage=state.var_usage,
    )


def _unread_answer_members(state: ResolveState) -> tuple[list[str], tuple[RoleAxis, ...]]:
    """The members a flagged match of the elected answer names when no read served it: those
    the settled bindings select, not the family (`_members_holding_bound_values`), with the
    open axes they declare a role on (`_declined_open_roles`)."""
    holding = _members_holding_bound_values(
        state.shape_draft, bound_values_from_bindings(state.bindings)
    )
    return list(holding), _declined_open_roles(state, holding)


def _declined_open_roles(state: ResolveState, members: Mapping[str, dict]) -> tuple[RoleAxis, ...]:
    """The flow axes the frame asks every member of (`Frame.open_roles`) on which some of
    `members` (each with its arcs), else some member the declined shape stands for
    (`shape_population_arcs`), declares a role. No read settles them, so a recipient-only
    reading states no open creditor."""
    arcs = members or shape_population_arcs(state.shape_draft)
    declared = {
        axis_for_role(role) for facts in arcs.values() for role in read_declared_roles(facts)
    }
    return tuple(axis for axis in state.frame.open_roles.value if axis in declared)


async def _denominator_refusal(
    state: ResolveState,
    mat_result: _MaterialiseOutcome | None,
    *,
    members: Sequence[str] | None = None,
    open_roles: Collection[RoleAxis] = (),
) -> RegionResult:
    """Refuse ``denominator_not_available`` with the elected shape as the flagged match,
    naming the StatVars materialise elected (`mat_result`), else `members`, else every
    confirmed member of the shape, its level reading as nearest_real (`_level_relaxation`),
    and the denominator the question named beside the one the flagged match divides by. The
    flagged match keeps the open axes materialise confirmed the elected members declare
    (`Materialised.open_axes`, `NoDataDraft.open_axes`), else `open_roles`, so "external debt per capita by
    every creditor in Kenya" flags a by-creditor reading. A shape with no level relaxation,
    such as one that is already a level, carries no nearest_real: relaxing one of its
    bindings would offer the reading the gate refused."""
    if isinstance(mat_result, (Materialised, NoDataDraft)):
        members = mat_result.sv_dcids
        open_roles = mat_result.open_axes
    flagged = await _elected_flagged_match(state, members=members, open_roles=open_roles)
    named = state.named_denominator
    level = _level_relaxation(state)
    return state.no_data(
        "denominator_not_available",
        flagged_match=(flagged,),
        nearest_real=None if level is None else await _nearest_real_for(state, level=level),
        denominator=NoDataDenominator(
            asked=named.ref if named is not None else None,
            found=flagged.resolved.shape.measurement_denominator,
        ),
        llm_usage=state.var_usage,
    )


def _level_relaxation(state: ResolveState) -> tuple[ShapeDraft, Relaxation] | None:
    """The level reading of the elected shape, the relaxation that drops its denominator: a
    recalled shape with the same population type, measured property, stat type and
    constraint properties, no denominator arc, and the same qualifier. A ratio states no
    qualifier where its denominator carries the price basis (IDS's outstanding external
    debt as a share of GNI, beside the stock in current US dollars), so a ratio with no
    qualifier reads the one level whose qualifier is the only one its levels hold. `None`
    when the elected shape divides by nothing, or no recalled shape is its level."""
    elected = state.shape_draft
    denominator = elected.meas_denom_dcid
    if denominator is None or elected.meas_denom_unresolved:
        return None
    properties = _constraint_properties(elected)
    levels = [
        shape
        for shape in state.shapes_found
        if shape.meas_denom_dcid is None
        and not shape.meas_denom_unresolved
        and shape.pop_type_dcid == elected.pop_type_dcid
        and shape.meas_prop_dcid == elected.meas_prop_dcid
        and shape.stat_type_dcid == elected.stat_type_dcid
        and _constraint_properties(shape) == properties
    ]
    level = next((s for s in levels if s.meas_qual_dcid == elected.meas_qual_dcid), None)
    if level is None and elected.meas_qual_dcid is None:
        qualifiers = {s.meas_qual_dcid for s in levels}
        level = levels[0] if len(qualifiers) == 1 else None
    if level is None:
        return None
    return level, Relaxation(
        axis="what", property_dcid="measurementDenominator", dropped_value_dcids=[denominator]
    )


def _constraint_properties(shape: ShapeDraft) -> frozenset[str]:
    return frozenset(key.property_dcid for key in shape.slot_keys if key.property_dcid)


async def _nearest_real_for(
    state: ResolveState, *, level: tuple[ShapeDraft, Relaxation] | None = None
) -> tuple[NearestRealSpec, ...] | None:
    """Probe one-binding relaxations, with a limit of 2 suggestions, or, with `level`, the
    one level reading of the elected shape (`_level_relaxation`).

    Each relaxation is probed in the original answer's place context and each suggestion is
    read over its own (`SuggestionRead.place`): the preview binds the places the
    suggestion's StatVars settle on, and a suggestion whose scope warning the shape in play
    does not hold carries it stamped with its `spec_id`. Fetch observations for survivors
    only on the buffered path. Return ``None`` when no relaxation finds data.
    """
    near_specs, near_warnings, near_reads, near_presence = await asyncio.to_thread(
        _suggest_nearest_real,
        graph=state.graph,
        sdmx=state.sdmx,
        shape_draft=state.shape_draft,
        bindings=state.bindings,
        place=state.place,
        presence=state.default_presence,
        date_request=state.date_request,
        deadline=state.deadline,
        include_option_coverage=state.include_option_coverage,
        materialise_graph=state.materialise_graph,
        role_class_walks=state.role_class_walks,
        query_entity_text_by_dcid=state.entity_label_text_by_dcid,
        # Preserve qualifier, measure, and default provenance on untouched bindings.
        text_recognized_properties=state.text_recognized_properties,
        grouped_properties=state.grouped_properties,
        measure_elected_properties=state.measure_elected_properties,
        defaulted_properties=state.defaulted_properties,
        named_groups=request_named_groups(state, state.place.roles),
        coverage=state.coverage,
        unsettled=state.unsettled_place,
        pin_served_provenance=state.pin_served_provenance,
        level=level,
    )
    if near_presence is not None:
        known = state.default_presence
        state.default_presence = near_presence if known is None else known.merged(near_presence)
    state.warnings.extend(
        warning
        for warning in near_warnings
        if warning.code != MEMBERSHIP_UNVERIFIED
        or all(held.code != MEMBERSHIP_UNVERIFIED for held in state.warnings)
    )
    if not near_specs:
        return None
    stamp_offered_place_warnings(
        state.warnings,
        [(spec.spec_id, near_reads[spec.spec_id].place) for spec in near_specs],
        departure_facts=state.departure_facts,
    )
    # Relaxed specs were reconfirmed against the graph. Fetch previews only when requested
    # on the buffered path. Nearest-real suggestions have no SSE observation event.
    if not state.include_observations or state.stream_observations:
        return tuple(near_specs)
    previews: list[tuple[int, dict[str, list[str]]]] = []
    for index, near_spec in enumerate(near_specs):
        place = near_reads[near_spec.spec_id].place
        obs_entities, obs_entities_warn = _observation_entities(
            state.shape_draft,
            destination_dcids=place.probe_places,
            destination_dcid=place.roles.obs_destination_dcid,
            origin_dcids=place.roles.origin_dcids,
        )
        if obs_entities_warn is not None and obs_entities_warn not in state.warnings:
            state.warnings.append(obs_entities_warn)
        if obs_entities:
            previews.append((index, obs_entities))
    results = await asyncio.gather(
        *[
            asyncio.to_thread(
                _build_observations,
                near_reads[near_specs[index].spec_id].plan,
                [sv.ref.ref for sv in near_specs[index].resolved.members],
                obs_entities,
                graph=state.graph,
                materialise_graph=state.materialise_graph,
                materialise_sdmx=state.materialise_sdmx,
                sdmx=state.sdmx,
                rate_cache=state.rate_cache,
                coverage=state.coverage,
                member_facts=near_reads[near_specs[index].spec_id].member_facts,
            )
            for index, obs_entities in previews
        ]
    )
    for (index, _), result in zip(previews, results, strict=True):
        near_specs[index] = cast(
            NearestRealSpec, _attach_observations(near_specs[index], result, state.warnings)
        )
    return tuple(near_specs)
