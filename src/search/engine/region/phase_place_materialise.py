"""Assign place sets to roles, settle them, and materialise the elected shape.

Handle no-data results after earlier phases fall through.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final, cast

import dc_graph
from dfo_platform.executor import submit_with_context
from dfo_platform.trace import get_trace_id
from materialise.place_scope import DefaultPresence
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import EngineGraphClient, force_full_facets
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.config import SEARCH_MAX_UNIT_ALTERNATIVES, settings
from search.engine.envelope import _emit_step, now_ms
from search.engine.errors import FamilyNotResolvableError, GraphInfraError, RoleSignatureError
from search.engine.interpret import PinnedResolved, Recall
from search.engine.place import earth_countries, read_default_presence
from search.engine.quantity import PER_PERSON_FORMS, fulfils
from search.engine.region.alternate_guard import AlternateJudge, query_pins
from search.engine.region.availability import _shape_group_key
from search.engine.region.candidate_options import _elected_sibling_warning
from search.engine.region.candidates import (
    _Alternate,
    _Candidate,
    _materialise_alt_shapes,
    alternate_representative,
    walk_alternate,
)
from search.engine.region.codes import (
    MEMBERSHIP_UNVERIFIED,
    SEAM_OFF_INFO_CODE,
    _elect_no_data_reason,
)
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
from search.engine.region.near_neighbor import (
    _representative_score_is_weak,
    _shapes_semantically_distinct,
)
from search.engine.region.observations.dispatch import _attach_observations
from search.engine.region.observations.entities import _observation_entities, open_role_axis
from search.engine.region.observations.fetch import _build_observations
from search.engine.region.phase_intake import not_admitted_entity
from search.engine.region.phase_rank_bind import _raw_count_is_pending_joint_slice
from search.engine.region.phase_shared import (
    _admitted_qualifier_properties,
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
    bind_alternate_place_roles,
    bind_place_roles,
    bound_after_election,
    first_spellings,
    names_a_place,
    request_named_groups,
)
from search.engine.region.result import FacetAxisFilter, RegionResult
from search.engine.region.sentence_events import _build_sentence_event
from search.engine.region.state import (
    AlternateUnevaluatedReason,
    ResolveState,
    for_shape,
    served_stat_vars,
)
from search.engine.region.unrealisable import unbind_unrealisable
from search.engine.retrieval.bound_values import (
    RoleKeys,
    bound_values_from_bindings,
    elected_role_class_shape,
    shape_query,
)
from search.engine.retrieval.facts import read_constraints
from search.engine.retrieval.retrieve import (
    BoundValuesUnrealisable,
    Materialised,
    MaterialisedCandidates,
    NoDataDraft,
    UnsupportedDraft,
)
from search.engine.retrieval.standard import FacetBatch
from search.engine.shape import (
    ShapeDraft,
    is_role_bearing,
)
from search.engine.unit.base_year import declared_base_year
from search.engine.unit.bind import UnitDefaultPolicy, bind_unit_text
from search.engine.unit.refs import satisfies as _unit_satisfies
from search.models import (
    FacetBasis,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    NearestRealSpec,
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
    """
    _note_named_places(state)

    bound = bind_place_roles(state, state.shape_draft)
    if bound == "role_incompatible":
        return await _role_incompatible_no_data(state)
    if bound == "role_not_resolved":
        return await _role_not_resolved_no_data(state)
    state.elected_place_roles = bound
    # The bound roles elect one role class among the shape's members, and every later
    # step reads the shape narrowed to it. The mint's walk settles the members off the
    # event loop, for any shape with a member graph to walk. Members it does not settle
    # still answer, under a warning.
    role_keys = bound.role_keys
    role_query = shape_query(
        state.shape_draft, state.bindings, role_keys.bound, open_role_keys=role_keys.open
    )
    recalled = state.shape_draft
    resolution = elected_role_class_shape(state.shape_draft, role_query)
    state.shape_draft = resolution.value
    # The alternates and siblings start their walks now, beside the elected shape's own, under
    # the plan the recalled election gives. A walk that changes the elected shape makes the
    # plan again, and a walk the new plan needs and this one did not starts then.
    planned, plan = state.shape_draft, _plan_alternates(state)
    if not names_a_place(state):
        _start_countries_listing(state, plan)
    if state.role_class_walks is not None:
        try:
            resolution = await asyncio.to_thread(
                elected_role_class_shape,
                state.shape_draft,
                role_query,
                walks=state.role_class_walks,
                arcs_graph=state.graph,
                deadline=state.deadline,
            )
        except RoleSignatureError:
            # The mint refuses this identity for role sets its walk cannot order.
            return state.no_data("role_signature_incompatible", llm_usage=state.var_usage)
        except FamilyNotResolvableError:
            # The mint refuses this identity for a member at fault, so its series are not
            # resolved.
            return state.no_data("variable_not_resolved", llm_usage=state.var_usage)
        state.shape_draft = resolution.value
    if not resolution.authority.settled:
        state.warnings.append(membership_unverified_warning())
    state.walked = resolution.walked
    # The place context below is built for the shape the election returned.
    bound = bound_after_election(state, recalled, bound, state.shape_draft)
    if bound == "role_incompatible":
        return await _role_incompatible_no_data(state)
    if bound == "role_not_resolved":
        return await _role_not_resolved_no_data(state)
    if state.shape_draft is not planned:
        plan = _plan_alternates(state)
    (unsettled,) = await _place_contexts(
        state,
        [(state.shape_draft, state.bindings, bound)],
        group=[*plan.alt_indices, *state.reelection_siblings, *plan.cascade_indices],
    )
    try:
        place = unsettled.unread()
    except PlaceSetTooLargeError as exc:
        return _too_many_places(state, exc)
    except PlaceSetNotAdmittedError as exc:
        return _not_admitted_places(state, exc)
    _install_shape(state, state.shape_draft, state.bindings, place, unsettled)
    if place.default_scope is None:
        state.alternates_presence = _start_alternates_presence_read(state, plan)
    return None


def _plan_alternates(state: ResolveState) -> _AdmissionPlan:
    """Bind every alternate's places (`_bind_alternates`), choose the re-election siblings
    (`_reelection_siblings`), apply the admission guards and budget (`_plan_admission`), and
    start the walk of each alternate and sibling the plan names (`_start_alternate_walks`).
    Makes no request of its own.

    An alternate is walked at the representative it takes beside an elected shape with data,
    the case in which admission offers it; a sibling at the one it takes beside an elected
    shape without data, the only case in which re-election runs."""
    state.alternate_place_roles, state.unbindable_alternates = _bind_alternates(
        state,
        [
            (i, shape)
            for i, shape in enumerate(state.capped_shapes)
            if i != state.bind_result.chosen_shape_index
        ],
    )
    state.reelection_siblings = _reelection_siblings(state)
    plan = _plan_admission(state)
    _start_alternate_walks(
        state,
        {i: state.capped_shapes[i] for i in [*plan.alt_indices, *plan.cascade_indices]},
        elected_has_data=True,
    )
    _start_alternate_walks(state, _sibling_shapes_of(state), elected_has_data=False)
    return plan


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
) -> None:
    """Make `shape` the shape in play, with its property bindings and its place context.

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
    state.bindings = with_where(bindings, place)
    state.place = place
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


def _group_stat_vars(
    state: ResolveState, group: Sequence[int], shapes: Sequence[ShapeDraft] = ()
) -> list[str]:
    """The StatVars a default-scope presence read covers for `shapes`, then for the capped
    shapes at `group`'s indices (`default_scope_stat_vars`). `shapes` holds shapes as the
    role-class election returned them, which the capped menu does not."""
    return default_scope_stat_vars(
        [
            *shapes,
            *(
                state.capped_shapes[i]
                for i in dict.fromkeys(group)
                if 0 <= i < len(state.capped_shapes)
            ),
        ]
    )


def _lacks_presence(
    state: ResolveState, group: Sequence[int], shapes: Sequence[ShapeDraft] = ()
) -> bool:
    """Whether `state.default_presence` lacks a subject-only StatVar of `shapes` or of the
    capped shapes at `group`'s indices."""
    known = state.default_presence
    return known is None or any(
        sv not in known.scopes for sv in _group_stat_vars(state, group, shapes)
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
    places = [
        UnsettledPlace(
            shape,
            roles,
            stat_vars=tuple(served_stat_vars(shape, bindings, roles.role_keys))
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
    takers = [shape for shape, _, roles in shapes if takes_default_scope(shape, roles)]
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
    requirement (`state.quantity_requirement`, which the materialise check also reads) and
    the kinds of the denominators read before bind (`state.denominator_kinds`). A judge this
    pass built for the same elected shape under equal pins is reused
    (`state.alternate_judges`)."""
    pins = query_pins(
        state.variable,
        bindings=state.bindings,
        engine_authored=state.measure_elected_properties | state.defaulted_properties,
        qualifier_matches=for_shape(state.query_qualifier_matches, state.shape_draft),
        qualifier_readings=state.query_qualifier_readings,
        unread_price_basis=state.unread_price_basis,
        elected_has_data=elected_has_data,
        denominator_kinds=state.denominator_kinds,
        quantity=state.quantity_requirement,
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
    the admission guards alone, before any read is made for them."""

    # Why each alternate the guards reject is left out, by capped-shape index: its roles
    # cannot bind the query's places, it cannot take the default scope the query gives it,
    # or it contradicts the query's wording.
    inadmissible: dict[int, str]
    # The flagged alternates that pass the guards, and those within the candidate budget.
    eligible: list[int]
    alt_indices: list[int]
    # The shapes a widened search adds when a tie fired and its alternate passes the guards,
    # and those within the budget the alternates leave. Empty otherwise.
    eligible_cascade: list[int]
    cascade_indices: list[int]


def _plan_admission(state: ResolveState) -> _AdmissionPlan:
    """Apply the admission guards and the candidate budget to the alternates of the shape
    in play, making no request.

    A shape that is not a reading of the shape in play (`alternate_guard`) is rejected
    before budget or materialisation; registry twins pass for equivalent-series handling.
    Every registry twin of the elected reading on the menu
    (`AlternateJudge.twin_alternates`) is admitted first whether or not bind nominated it.
    Every capped shape is compared with the shape in play, so the guard covers model
    nominees, tie proofs and the widening cascade. A shape whose roles the query's places
    cannot bind, or that cannot take the default scope a query naming no place gives it
    (`state.unbindable_alternates`), is rejected the same way, twins included. The plan is
    made twice: before any read, and once the shape in play's data is known
    (`admit_alternates`), when a lower-ranked price basis of the same family is refused too.

    A confirmed tie remains ambiguous when its alternate has no data, and the search then
    widens beyond `SEARCH_NEAR_MARGIN` to semantically distinct shapes. The plan lists them
    whenever a tie fired and its alternate passes the guards; whether the search runs is
    decided once the elected shape and that alternate are materialised (`admit_alternates`).
    """
    chosen = state.bind_result.chosen_shape_index
    budget = max(0, settings().max_candidates - 1)
    bindable = [(i, state.capped_shapes[i]) for i in state.alternate_place_roles]
    judge = _guard_judge(state, elected_has_data=state.elected_has_data)
    inadmissible = {**state.unbindable_alternates, **judge.inadmissible(bindable)}
    # Every registry twin of the elected reading on the menu is admitted first, the
    # preferred one leading, nominated or not, so the equivalent-series diversion serves the
    # preferred twin and lists the others whatever bind nominated.
    twins = judge.twin_alternates([(i, shape) for i, shape in bindable if i != chosen])
    eligible = [
        i
        for i in dict.fromkeys([*twins, *(state.bind_result.also_plausible or [])])
        if 0 <= i < len(state.capped_shapes) and i != chosen and i not in inadmissible
    ]
    alt_indices = [] if state.force_single_shape_path else eligible[:budget]
    eligible_cascade: list[int] = []
    fired = state.near_neighbor_fired_index
    if fired is not None and fired not in inadmissible:
        # Exclude guard-rejected candidates from every source, including cascade-only shapes.
        tried = {chosen, *alt_indices, *inadmissible}
        eligible_cascade = [
            i
            for i, cand in enumerate(state.capped_shapes)
            if i not in tried and _shapes_semantically_distinct(state.shape_draft, cand)
        ]
    return _AdmissionPlan(
        inadmissible=inadmissible,
        eligible=eligible,
        alt_indices=alt_indices,
        eligible_cascade=eligible_cascade,
        cascade_indices=eligible_cascade[: max(0, budget - len(alt_indices))],
    )


@dataclass(frozen=True)
class _WalkedAlternate:
    """An alternate as the role-class election returned it (`elect_alternate`), with the
    query's places assigned to that shape (`place_roles.bound_after_election`)."""

    alternate: _Alternate
    roles: PlaceAssignment

    @property
    def shape(self) -> ShapeDraft:
        return self.alternate.shape


_WalkKey = tuple[int, tuple[str, ...], tuple[object, ...], str]


def _walk_key(state: ResolveState, index: int, shape: ShapeDraft, rep_sv: str) -> _WalkKey:
    """The memo key of a walk: everything `walk_alternate` reads from the request. That is
    the alternate, by capped-shape index, its members and the role keys its places are
    assigned under, and the representative its bindings come from. The dataset it names is
    its shape's (`ShapeDraft.named_dataset`), or the one the part's source scope leaves
    serving under those bindings, whatever bindings the shape in play holds. The scope is
    the leg's, fixed once vetting runs."""
    role_keys = state.alternate_place_roles[index].role_keys
    places = (tuple(sorted(role_keys.bound)), tuple(sorted(role_keys.open)))
    return (index, tuple(shape.sv_arc_facts or {}), places, rep_sv)


def _walk_args(state: ResolveState, index: int, shape: ShapeDraft, rep_sv: str) -> dict:
    return {
        "shape": shape,
        "role_keys": state.alternate_place_roles[index].role_keys,
        "rep_sv": rep_sv,
        "graph": state.graph,
        "deadline": state.deadline,
        "role_class_walks": state.role_class_walks,
        "scope": state.scope,
    }


def _start_alternate_walks(
    state: ResolveState, shapes: Mapping[int, ShapeDraft], *, elected_has_data: bool
) -> None:
    """Start the walk of each of `shapes`, by capped-shape index and narrowed to the members
    it is offered through (`_offered_shapes`), at the representative it takes under
    `elected_has_data` (`alternate_representative`), on `_WALK_POOL`, and keep
    each in `state.alternate_walks` under its `_walk_key`. A walk already kept is not
    started again. With no walks to run (`state.role_class_walks` is `None`) the election
    makes no request, so nothing is started and `_walk` elects in place."""
    if state.role_class_walks is None:
        return
    for index, shape in _offered_shapes(state, shapes).items():
        rep_sv = alternate_representative(
            shape, elected_bindings=state.bindings, elected_has_data=elected_has_data
        )
        if rep_sv is None:
            continue
        key = _walk_key(state, index, shape, rep_sv)
        if key in state.alternate_walks:
            continue
        state.alternate_walks[key] = _submit_walk(state, index, shape, rep_sv)


def _submit_walk(state: ResolveState, index: int, shape: ShapeDraft, rep_sv: str) -> Future:
    """Queue the walk of `shape` at `rep_sv` on `_WALK_POOL` (`walk_alternate`)."""
    return submit_with_context(
        _WALK_POOL, walk_alternate, **_walk_args(state, index, shape, rep_sv)
    )


async def _walk(
    state: ResolveState, index: int, shape: ShapeDraft, *, elected_has_data: bool
) -> _Alternate | None:
    """The walk of `shape` at the representative it takes under `elected_has_data`: the one
    `_start_alternate_walks` started, awaited however long it waits in `_WALK_POOL`'s queue,
    else one queued there now. With no walks to run the election makes no request and runs
    in place."""
    rep_sv = alternate_representative(
        shape, elected_bindings=state.bindings, elected_has_data=elected_has_data
    )
    if rep_sv is None:
        return None
    if state.role_class_walks is None:
        return walk_alternate(**_walk_args(state, index, shape, rep_sv))
    key = _walk_key(state, index, shape, rep_sv)
    started = state.alternate_walks.get(key)
    if started is None:
        started = state.alternate_walks[key] = _submit_walk(state, index, shape, rep_sv)
    return await asyncio.wrap_future(started)


async def _walked_alternates(
    state: ResolveState, shapes: Mapping[int, ShapeDraft], *, elected_has_data: bool
) -> dict[int, _WalkedAlternate]:
    """Elect each alternate's role class before anything is read for its places, and bind
    the query's places to the shape the election returns. Keyed by capped-shape index, in
    `shapes` order, without the alternates the election or the binding drops. Each shape is
    walked through the members it is offered through (`_offered_shapes`).

    The mint's walk can replace a role-bearing shape with a subject-only family, or the
    reverse, so the presence read, the place context and the probe places all belong to the
    shape this returns. Every alternate is walked, whether or not it declares a flow role,
    so an offered shape lists the members its own resubmit serves. The walks run beside one
    another off the event loop. A shape keeps the binding `_phase_place_bindings` made
    unless the election turns it from role-bearing to subject-only or the reverse
    (`place_roles.bound_after_election`); a shape whose new binding refuses is dropped.
    """
    shapes = _offered_shapes(state, shapes)
    elected = await asyncio.gather(
        *[_walk(state, i, shape, elected_has_data=elected_has_data) for i, shape in shapes.items()]
    )
    walked: dict[int, _WalkedAlternate] = {}
    for (i, shape), alternate in zip(shapes.items(), elected, strict=True):
        if alternate is None:
            continue
        roles = bound_after_election(state, shape, state.alternate_place_roles[i], alternate.shape)
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
    one, so it starts here, and admission awaits it (`_read_alternates_presence`). It covers
    the alternates and the shapes a widened search would add. When the elected result ends
    the request, the read is left to finish on its pool and nothing waits for it. With
    every worker of the pool busy it is not started, and admission tries it again.
    """
    if not any(
        takes_default_scope(state.capped_shapes[i], state.alternate_place_roles[i])
        for i in plan.alt_indices
    ):
        return None
    return _submit_alternates_presence_read(
        state, _group_stat_vars(state, [*plan.alt_indices, *plan.cascade_indices])
    )


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


async def _await_alternates_presence(
    state: ResolveState, read: Future[DefaultPresence]
) -> AlternateUnevaluatedReason | None:
    """Await `read` and merge its presence into the request's, or return why it did not
    complete. The read was made over the presence known when it was submitted, so a read
    made since (a re-election's) keeps what it added. The read serves alternates only, so its
    failure leaves them unevaluated and the elected answer stands."""
    try:
        presence = await asyncio.wrap_future(read)
        known = state.default_presence
        state.default_presence = presence if known is None else known.merged(presence)
    except (TimeoutError, dc_graph.GraphDeadlineExceededError):
        return "timed_out"
    except (GraphInfraError, dc_graph.GraphReadError, dc_graph.GraphCapExceededError):
        return "upstream_error"
    return None


async def _read_alternates_presence(
    state: ResolveState,
    shapes: Mapping[int, tuple[ShapeDraft, PlaceAssignment]],
    *,
    group: Sequence[int],
) -> dict[int, AlternateUnevaluatedReason]:
    """Read the presence the alternates in `shapes` need for their default scopes, and
    return the indices left without it, in order, each with the reason.

    `shapes` holds each alternate by capped-shape index as the role-class election returned
    it, with the places bound to that shape (`_walked_alternates`), so the read covers the
    StatVars the alternate is probed through. The read started beside a role-bearing
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
    takers = {i: shape for i, (shape, roles) in shapes.items() if takes_default_scope(shape, roles)}
    if not takers:
        return {}
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
        (i for i, shape in takers.items() if _lacks_presence(state, [], [shape])), reason
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


def _read_settles_a_set(state: ResolveState) -> bool:
    """Whether the elected shape's own read decides the reading of one of its place sets
    (`UnsettledPlace.decided`)."""
    return state.unsettled_place is not None and not state.unsettled_place.decided


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


async def _emit_bind_sentence(state: ResolveState) -> None:
    """Build and emit the bind-stage sentence of a measure-elected shape. It names the
    elected measure family and leaves time and source blank; a definite answer's final
    sentence fills them."""
    if state.emit is None or state.measure_elected_partition is None:
        return
    sentence_event = await _build_sentence_event(
        state.shape_draft,
        state.bindings,
        state.place.roles.roles,
        variable_index=state.variable_index,
        date_request=state.date_request,
        open_axes=state.place.roles.open_axes,
        graph=state.graph,
        query_entity_text_by_dcid=state.entity_label_text_by_dcid,
        measure_partition=state.measure_elected_partition,
        named_groups=request_named_groups(state, state.place.roles),
        explicit_facets=(
            # A bound outcome guarantees a unit request. The type checker cannot infer it.
            {
                "unit": IdentityFacetValueRequest(
                    request=cast(IdentityUnitRequest, state.unit_request)
                )
            }
            if state.bound_unit_outcome is not None
            else None
        ),
    )
    await state.emit(sentence_event)


async def _run_materialise_task(state: ResolveState) -> _MaterialiseOutcome:
    """Run materialisation off-thread and time it, and install the place context the
    elected shape's place sets settle to (`_materialise_task`). A measure-elected sentence
    is built and emitted while it runs, or once it has run when the read decides which
    places the sentence names. Over a graph that can declare a default facet the pipeline
    holds it until the part's answer stands, so it never precedes a declared-default refusal
    from this or a later step (`pipeline.resolve_variable`). Bound values that together match
    no member lose a minimal set and the shape is resolved again
    (`unrealisable.unbind_unrealisable`) until its bindings can be read.

    Raises `PlaceSetTooLargeError` when the places the sets settle to exceed the bound on
    the places a Spec binds."""
    t0 = now_ms()
    settles = _read_settles_a_set(state)
    materialise_task = _materialise_task(state)
    try:
        if not settles:
            # Build and emit while the materialise call runs.
            await _emit_bind_sentence(state)
        place, mat_result = await materialise_task
        while isinstance(mat_result, BoundValuesUnrealisable):
            unbind_unrealisable(state, mat_result)
            materialise_task = _materialise_task(state)
            place, mat_result = await materialise_task
        if place != state.place:
            _install_shape(state, state.shape_draft, state.bindings, place, state.unsettled_place)
        if settles:
            await _emit_bind_sentence(state)
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


def _is_qualifier_variant_sibling(a: ShapeDraft, b: ShapeDraft) -> bool:
    """Match shapes differing only in ``meas_qual_dcid``, with stat type and denominator equal."""
    return (
        not _shapes_semantically_distinct(a, b)
        and a.stat_type_dcid == b.stat_type_dcid
        and a.meas_qual_dcid != b.meas_qual_dcid
        and a.meas_denom_dcid == b.meas_denom_dcid
        and a.meas_denom_unresolved == b.meas_denom_unresolved
    )


def _members_holding_bound_values(
    shape: ShapeDraft, bound_values: Mapping[str, list[str]]
) -> dict[str, dict]:
    """Return the members of `shape` that assert every property `bound_values` binds, at
    a bound value, with their arcs.

    A qualifier variant measures the elected quantity only when it keeps the elected
    bindings: nominal GDP beside inflation-adjusted GDP both hold
    ``activitySource=GrossDomesticProduction``, and GNI at purchasing power parity holds
    another value. No bound value admits every member.
    """
    return {
        sv: arcs
        for sv, arcs in (shape.sv_arc_facts or {}).items()
        if all(read_constraints(arcs).get(prop) in values for prop, values in bound_values.items())
    }


def _reelection_siblings(state: ResolveState) -> list[int]:
    """The capped-shape indices of the siblings zero-observation re-election would try.

    Candidates are unflagged qualifier variants with scores inside ``weak_score_threshold -
    relevance_threshold`` of a scored elected shape, role-bearing or not, whose score is
    neither the sentinel ``1.0`` nor weak. This local margin does not change the general
    near-neighbour gate. Siblings whose roles the query's places bind and that can take the
    scope the query gives them (``state.alternate_place_roles``) pass the alternate guard
    (`AlternateJudge.inadmissible`) before the ``max_candidates - 1`` cap. The elected
    shape has no data here, so no default price basis is imposed. They depend on the
    elected shape and the bind step alone, so they are known before any read.
    """
    _elected_score = state.shape_draft.representative_score
    if (
        _elected_score is None
        or _elected_score == 1.0
        or _representative_score_is_weak(_elected_score)
    ):
        return []
    _already_flagged = {
        state.bind_result.chosen_shape_index,
        *(state.bind_result.also_plausible or []),
    }
    _sibling_score_margin = settings().weak_score_threshold - settings().relevance_threshold
    _elected_bound_values = bound_values_from_bindings(state.bindings)
    _sibling_indices = [
        i
        for i, cand in enumerate(state.capped_shapes)
        if i not in _already_flagged
        and i in state.alternate_place_roles
        and _is_qualifier_variant_sibling(state.shape_draft, cand)
        and _members_holding_bound_values(cand, _elected_bound_values)
        and cand.representative_score is not None
        and _elected_score - cand.representative_score <= _sibling_score_margin
    ]
    if _sibling_indices:
        _inadmissible = _guard_judge(state, elected_has_data=False).inadmissible(
            [(i, state.capped_shapes[i]) for i in _sibling_indices]
        )
        if _inadmissible:
            _sibling_indices = [i for i in _sibling_indices if i not in _inadmissible]
    return _sibling_indices[: max(0, settings().max_candidates - 1)]


def _sibling_shapes_of(state: ResolveState) -> dict[int, ShapeDraft]:
    """Each re-election sibling (`state.reelection_siblings`) by capped-shape index, holding
    the members it is probed and answered through: those that hold the elected bindings'
    values (`_members_holding_bound_values`)."""
    elected_bound_values = bound_values_from_bindings(state.bindings)
    return {
        i: dataclasses.replace(
            state.capped_shapes[i],
            sv_arc_facts=_members_holding_bound_values(
                state.capped_shapes[i], elected_bound_values
            ),
        )
        for i in state.reelection_siblings
    }


async def _reelect_zero_observation_sibling(
    state: ResolveState, mat_result: Materialised
) -> Materialised:
    """Try the nearby qualifier siblings (`state.reelection_siblings`) when the elected
    shape has no observations at its places (``has_data=False``,
    ``available_at_place=False``).

    Each sibling's members are settled first by the mint's walk (`_walked_alternates`), so a
    sibling that answers lists the members its own resubmit serves, and
    ``_materialise_alt_shapes`` then prewarms facets and resolves the siblings. Each sibling
    settles its own place sets on its own read, for the shape the election returned
    (`_place_contexts`, `place_context.settle_sets`), or takes the default scopes of its own
    StatVars when it is subject-only and the query names no place. The elected shape's
    presence read covers those scopes (`_phase_place_bindings`) unless the walk replaced the
    sibling's members, so no read is made here otherwise. Siblings with separate probe
    places are read concurrently, so their facet reads take one serial step between them.

    One data-bearing subject sibling is installed in place of the elected shape with its
    bindings and place context (`_install_shape`), replaces the elected result, and adds a
    warning after its place warnings. A winner whose members the walk left unsettled states
    it once its result is read (`interpret_materialised`). Multiple or role-bearing results join
    ``also_plausible`` and are kept as `state.sibling_candidates`, which candidate admission
    takes in place of materialising them (`admit_alternates`), and the elected shape stays
    in play with its own context. Blocked or empty probes leave the result and the place
    context unchanged.
    """
    _sibling_shapes = _sibling_shapes_of(state)
    # A sibling's members are settled before its places, presence and context, which belong
    # to the shape it is answered through.
    _walked = await _walked_alternates(state, _sibling_shapes, elected_has_data=False)
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
    _scope_results: list[tuple[list[_Candidate], UnsupportedDraft | None]] = await asyncio.gather(
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
    # The role-class election can replace a sibling's shape, so its group key, which the
    # election leaves unchanged, maps a result back to its menu index.
    _index_by_group = {_shape_group_key(_walked[i].shape): i for i in _walked}
    # An over-cap sibling is unevaluated. Keep the elected no-observations result rather
    # than treating that sibling as disproven or surfacing a cap refusal.
    if any(_sibling_blocking is not None for _, _sibling_blocking in _scope_results):
        return mat_result
    _sibling_candidates = [
        cand for _scope_candidates, _ in _scope_results for cand in _scope_candidates
    ]
    if not _sibling_candidates:
        return mat_result
    # ``state.reelection_siblings`` ascend, so this restores menu order across scope groups.
    _sibling_candidates.sort(key=lambda cand: _index_by_group[_shape_group_key(cand.shape)])

    _reelectable = [cand for cand in _sibling_candidates if not is_role_bearing(cand.shape)]
    if len(_sibling_candidates) == 1 and len(_reelectable) == 1:
        (_winner,) = _reelectable
        _no_data_label = state.shape_draft.label
        _install_shape(state, _winner.shape, _winner.bindings, _winner.place)
        state.walked = _winner.walked
        state.mat_result = _winner.mat
        mat_result = _winner.mat
        state.warnings.append(
            _elected_sibling_warning(no_data=[_no_data_label], elected=[_winner.shape.label])
        )
    else:
        state.sibling_candidates = {
            _index_by_group[_shape_group_key(cand.shape)]: cand for cand in _sibling_candidates
        }
        state.bind_result.also_plausible = [
            *(state.bind_result.also_plausible or []),
            *state.sibling_candidates,
        ]
    return mat_result


async def _phase_materialise(state: ResolveState) -> RegionResult | None:
    """Materialise the elected shape, settling its place sets on that read, re-elect a
    zero-observation sibling, and apply per-capita denominator precedence."""
    try:
        mat_result = await _run_materialise_task(state)
    except PlaceSetTooLargeError as exc:
        return _too_many_places(state, exc)
    except PlaceSetNotAdmittedError as exc:
        return _not_admitted_places(state, exc)

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

    # A place-less empty probe proves a missing joint slice only when each same-head shape
    # loses at least two named qualifiers. Use the single-shape mint and retract near-neighbour
    # candidates and warnings.
    if (
        state.destination_dcid is None
        and isinstance(mat_result, Materialised)
        and not mat_result.has_data
        and mat_result.available_at_place is None
        and _raw_count_is_pending_joint_slice(
            state.shape_draft, state.eligible_shapes, state.variable
        )
    ):
        fired_index = state.near_neighbor_fired_index
        if fired_index is not None:
            state.bind_result.also_plausible = [
                index for index in (state.bind_result.also_plausible or []) if index != fired_index
            ]
        if state.near_neighbor_warning is not None:
            state.warnings.remove(state.near_neighbor_warning)
        state.near_neighbor_fired_index = None
        state.near_neighbor_warning = None
        state.force_single_shape_path = True

    # Preserve no-data precedence. ``no_matching_slice`` and ``variable_not_resolved`` win.
    # Otherwise an elected shape that does not divide by the count of persons the leg names
    # (`quantity.fulfils`: an amount, or a share of GNI, for "aid per capita") outranks
    # ``no_observations`` or base-measure data.
    per_person = state.quantity_requirement.narrowed(PER_PERSON_FORMS)
    if per_person.named and not fulfils(per_person, state.shape_draft, state.denominator_kinds):
        # Defer place-less, no-data materialised results so the pipeline can mint
        # ``no_matching_slice`` with its flagged match.
        _slice_mint_pending = (
            state.destination_dcid is None
            and isinstance(mat_result, Materialised)
            and not mat_result.has_data
        )
        _elected = _elect_no_data_reason(
            mat_result.reason if isinstance(mat_result, NoDataDraft) else None,
            "no_matching_slice" if _slice_mint_pending else None,
            fallback="denominator_not_available",
        )
        # Neither outcome reaches candidate fan-out, so retract its near-neighbour warning.
        if state.near_neighbor_warning is not None:
            state.warnings.remove(state.near_neighbor_warning)
        state.near_neighbor_fired_index = None
        state.near_neighbor_warning = None
        if _slice_mint_pending and _elected == "no_matching_slice":
            # The single-shape path attaches the flagged match after Spec assembly.
            state.force_single_shape_path = True
        else:
            return state.no_data(_elected, llm_usage=state.var_usage)

    return None


async def _nearest_real_for(state: ResolveState) -> tuple[NearestRealSpec, ...] | None:
    """Probe one-binding relaxations, with a limit of 2 suggestions.

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
        text_recognized_properties=_admitted_qualifier_properties(state.admitted_qualifiers),
        measure_elected_properties=state.measure_elected_properties,
        defaulted_properties=state.defaulted_properties,
        named_groups=request_named_groups(state, state.place.roles),
        coverage=state.coverage,
        unsettled=state.unsettled_place,
        pin_served_provenance=state.pin_served_provenance,
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
                [sv.ref.dcid for sv in near_specs[index].resolved.members],
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


async def _phase_no_data_answer(state: ResolveState) -> RegionResult | None:
    """Return the single-shape no-data result. ``None`` means materialisation found data.

    Attach nearest-real suggestions for ``no_observations``. Pass materialised results to
    definite assembly.
    """
    mat_result = state.mat_result

    if isinstance(mat_result, NoDataDraft):
        # Relax constraint bindings to suggest nearby data when observations are missing.
        nearest_real: tuple[NearestRealSpec, ...] | None = None
        if mat_result.reason == "no_observations":
            nearest_real = await _nearest_real_for(state)
        return RegionResult(
            variable_text=state.variable,
            status="no_data",
            specs=(),
            no_data_reason=mat_result.reason,
            warnings=tuple(state.warnings),
            departure_facts=dict(state.departure_facts),
            timing_by_step=dict(state.timing),
            settled_dataset=state.settled_dataset(),
            nearest_real=nearest_real,
            llm_usage=state.var_usage,
            no_data_role_axis=(
                open_role_axis(state.shape_draft, state.place.roles.resolved_entities)
                if mat_result.reason == "no_observations"
                else None
            ),
        )

    if isinstance(mat_result, UnsupportedDraft):
        return _unsupported(
            state,
            mat_result.code,
            pointer=mat_result.pointer,
            detail=mat_result.detail,
            llm_usage=state.var_usage,
            facets=mat_result.facets,
        )

    # Treat unexpected resolver outcomes as ``no_observations``.
    if not isinstance(mat_result, Materialised):
        return state.no_data("no_observations", llm_usage=state.var_usage)

    return None
