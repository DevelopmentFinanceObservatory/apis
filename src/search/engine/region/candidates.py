"""Build and collapse the ``also_plausible`` candidate fan-out.

Each candidate carries its own shape, bindings, materialised result and place context
(`_Candidate`). ``resolve_candidates`` returns ``None`` when no ambiguity survives.
Its sibling modules import helpers here, so it imports those modules locally
to avoid a cycle. Materialisation uses the shared retrieval interface.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from dc_graph import GraphCapExceededError, GraphDeadlineExceededError, GraphReadError
from materialise.errors import DatasetDisagreementError
from materialise.rate_tables import RateTableCache
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.config import settings
from search.engine.degraded_reads import DegradedReads, recording_degraded_reads
from search.engine.errors import FamilyNotResolvableError, GraphInfraError, RoleSignatureError
from search.engine.extract import DateRequest
from search.engine.place import ConsultedPresence
from search.engine.region.arcs import standard_bindings_from_arcs
from search.engine.region.codes import OBSERVATIONS_NOTHING_TO_PROBE
from search.engine.region.decisions import PlaceSetNotAdmittedError, PlaceSetTooLargeError
from search.engine.region.named_source import with_named_source
from search.engine.region.place_context import (
    PlaceContext,
    UnsettledPlace,
    settle_sets,
    settled_shape,
    unread_context,
    with_where,
)
from search.engine.region.result import RegionResult
from search.engine.region.state import (
    AlternateUnevaluatedReason,
    ResolveState,
    presence_covering,
    presence_lacking,
    presence_stat_vars,
)
from search.engine.retrieval.bound_values import (
    PartNotServed,
    RoleClassWalks,
    RoleKeys,
    WalkedMembers,
    applies_value,
    canonical_pin_bindings,
    elected_role_class_shape,
    entity_role_keys,
    named_scope,
    part_source,
    shape_for_part,
    shape_member_facts,
    shape_population_arcs,
    shape_query,
    without_values,
)
from search.engine.retrieval.facts import read_constraints
from search.engine.retrieval.representative import shape_agreement_representative
from search.engine.retrieval.retrieve import (
    BoundValuesUnrealisable,
    Materialised,
    MaterialisedCandidates,
    NoDataDraft,
    UnsupportedDraft,
    materialise,
)
from search.engine.retrieval.standard import FacetBatch, facets_at_own_places
from search.engine.shape import ShapeDraft, is_role_bearing
from search.engine.source_scope import SourceScope
from search.models import (
    CandidateBasis,
    ResolvedQualifier,
    SpecObservations,
    Warning,
    omitted_observations,
)
from series_coverage.shared import CoverageSnapshot

logger = logging.getLogger(__name__)

_UNIT_CHECK_NOT_PROBED: tuple[SpecObservations, Warning | None] = (
    omitted_observations(
        OBSERVATIONS_NOTHING_TO_PROBE,
        "the observation preview was not fetched: this candidate carries no data to "
        "check the requested unit against.",
    ),
    None,
)
"""The preview a streamed bound-unit survivor states when it is a flagged-unavailable
carrier (`available_at_place is False`), which holds no data to read
(`candidate_collapse._assemble_definite`). It carries a real, omitted `SpecObservations`, the
form `_build_observations` returns."""


@dataclass(frozen=True)
class _Candidate:
    """One shape offered for the answer: its bindings, its materialised result, the place
    context it is grounded and read over (`PlaceContext.settled` on `mat`), what the
    mint states of the members its walk settled (`MembershipResolution.walked`), and why the
    status rule offers it beside the elected shape (`status_rule`), `None` for the elected
    shape itself and for a reading that rides on a refusal, and the records of the words its
    own read leaves unread, its coverage records (`phase_election.judge_answer_coverage`), `None`
    for the elected shape, which reads the elected answer's (`requirements.judge`)."""

    shape: ShapeDraft
    bindings: list[SlotBindingDraft]
    mat: Materialised
    place: PlaceContext
    walked: WalkedMembers | None
    basis: CandidateBasis | None = None
    uncovered: tuple[ResolvedQualifier, ...] | None = None


def _prewarm_places(
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    role_keys: RoleKeys,
    members: Sequence[str],
    place: PlaceContext,
    presence: ConsultedPresence | None,
    named_place_dcids: Sequence[str],
    read_places: Sequence[str] = (),
) -> dict[str, tuple[str, ...]]:
    """The places a shape's facet prewarm reads each of `members` at: on a default scope,
    each member the presence read covers for `shape` under `bindings` and `role_keys`
    (`presence_stat_vars`), which the resolve reads at its own scope, at that scope
    (`ConsultedPresence.warming_places`), else every member at
    `read_places`, the places the settling of the shape's place sets reads
    (`UnsettledPlace.read_places`), else at the shape's probe places, else at every named
    place. A warm decides nothing, so it notes no scope."""
    scoped = place.presence_for(presence)
    if scoped is not None:
        return scoped.warming_places(
            members, own=set(presence_stat_vars(shape, bindings, role_keys))
        )
    entities = tuple(read_places or place.probe_dcids or named_place_dcids)
    return dict.fromkeys(members, entities)


@dataclass(frozen=True)
class _Alternate:
    """An alternate ready to materialise: its shape narrowed to the role class its entities
    elect, the constraint bindings of its representative StatVar, whether the mint's walk
    left its members unsettled (`MembershipAuthority.settled`), and what the mint states of
    the members it settled (`MembershipResolution.walked`)."""

    shape: ShapeDraft
    bindings: list[SlotBindingDraft]
    unverified: bool = False
    walked: WalkedMembers | None = None


def _stray_holdings(
    shape: ShapeDraft, stray_values: Mapping[str, frozenset[str]]
) -> dict[str, tuple[frozenset[str], list[str]]]:
    """For each property the bind drafted values on for a slot the elected reading lacks
    (`stray_values`), when some member of `shape` holds one: the member values that apply a
    drafted value, and the drafted values they apply, sorted (`applies_value`). With no
    value drafted it reads no member."""
    if not stray_values:
        return {}
    members = [read_constraints(arcs) for arcs in shape_population_arcs(shape).values()]
    holdings: dict[str, tuple[frozenset[str], list[str]]] = {}
    for prop, drafted in stray_values.items():
        held = {value for member in members if (value := member.get(prop)) is not None}
        applied = sorted(v for v in drafted if any(applies_value(v, value) for value in held))
        if applied:
            holding = frozenset(h for h in held if any(applies_value(v, h) for v in applied))
            holdings[prop] = (holding, applied)
    return holdings


def alternate_representative(
    shape: ShapeDraft,
    *,
    elected_bindings: list[SlotBindingDraft],
    elected_has_data: bool,
    stray_values: Mapping[str, frozenset[str]] | None = None,
) -> str | None:
    """The StatVar an alternate derives its constraint bindings from, or `None` when it has
    none.

    The shape arrives narrowed to the members the alternate guard offers it through, those
    matching the elected reading first (`alternate_guard.offered_alternates`), so its first
    member reads as the elected quantity. With elected data, a standard alternate needs a
    representative matching the elected bound values or a bare member. A role-bearing shape
    uses its first member because its constraint axes describe flows, and so does any shape
    when the elected shape has no data. On a property the bind drafted values on for a slot
    the elected reading lacks (`stray_values`), the member values that apply them count as
    bound, so a family holding one member per drafted value has a representative among
    them."""
    if is_role_bearing(shape) or not elected_has_data:
        return next(iter(shape_population_arcs(shape)), None)
    elected_bound_values: dict[str, list[str]] = {
        b.property_dcid: list(b.value_dcids)
        for b in elected_bindings
        if b.property_dcid and b.kind in ("value", "set") and b.value_dcids
    }
    for prop, (held, _drafted) in _stray_holdings(shape, stray_values or {}).items():
        elected_bound_values[prop] = sorted(held)
    return shape_agreement_representative(shape, elected_bound_values)


def _with_stray_values(
    bindings: list[SlotBindingDraft],
    shape: ShapeDraft,
    stray_values: Mapping[str, frozenset[str]],
) -> list[SlotBindingDraft]:
    """`bindings` with each slot on a property the bind drafted values on for a slot the
    elected reading lacks bound to the drafted values `shape`'s members apply: a set when
    they apply several, a value when one (`_stray_holdings`). The alternate then selects and
    reads exactly the members the question asks for."""
    holdings = _stray_holdings(shape, stray_values)
    return [
        binding.model_copy(
            update={
                "kind": "set" if len(drafted) > 1 else "value",
                "value_dcids": drafted,
            }
        )
        if (drafted := holdings.get(binding.property_dcid or "", (None, None))[1])
        else binding
        for binding in bindings
    ]


def walk_alternate(
    shape: ShapeDraft,
    role_keys: RoleKeys,
    rep_sv: str,
    *,
    graph: EngineGraphClient,
    deadline: float | None = None,
    role_class_walks: RoleClassWalks | None = None,
    scope: SourceScope | None = None,
    stray_values: Mapping[str, frozenset[str]] | None = None,
    bindings: Sequence[SlotBindingDraft] | None = None,
) -> _Alternate | None:
    """Give an alternate the constraint bindings of `rep_sv` and narrow it to the role class
    its entities elect, or return `None` when the mint refuses its identity or no table the
    part may answer the alternate from holds a row for those bindings
    (`bound_values.PartNotServed`), so an alternate is never served from a table outside
    the part's source scope or from a table of another family.

    The chosen shape alone receives model bindings. An alternate derives constraint bindings
    from its representative StatVar (`alternate_representative`), except on a property the
    bind drafted values on for a slot the elected reading lacks (`stray_values`), which it
    binds to the drafted values its members apply (`_with_stray_values`). A breakdown of
    the elected shape (`ResolveState.breakdown_rivals`) passes its own `bindings` instead.
    The alternate names the dataset its shape names (`with_named_source`), as the chosen
    shape does, so its identity is the one a question electing it mints. When the shape names none, the alternate names the
    dataset the part, under the source scope `scope`, is served from under those
    bindings and its role keys, where its unnamed identity would resolve to another at the
    mint (`part_source`), as the elected shape is named under the model's bindings. Its
    bindings on the invariants
    of the dataset its family elects take their canonical form as the chosen shape's do
    (`canonical_pin_bindings`), so its slots state what its canonical identity binds.

    ``role_keys`` are the keys the alternate's own identity binds the query's places under
    and the axes it leaves open (`PlaceAssignment.role_keys`). They elect one role class
    among its members (``elected_role_class_shape``), settled by the mint's walk when
    ``role_class_walks`` is given, whether or not the alternate declares a flow role, as the
    elected shape's is. The election reads no place, so it runs before the places settle. An
    alternate whose members the walk does not settle is marked `unverified`, and its result
    carries ``membership_unverified``. A walk that meets role sets the election cannot
    order, or a member at fault, drops the alternate, as the mint refuses its identity. The
    walk can change the roles the shape declares, so the caller assigns the returned
    shape's places (`place_roles.bound_after_election`) and settles them after this call.
    """
    served = _served_alternate(
        shape, role_keys, rep_sv, scope=scope, stray_values=stray_values or {}, bindings=bindings
    )
    if served is None:
        return None
    shape, bindings = served
    try:
        resolution = elected_role_class_shape(
            shape,
            shape_query(shape, bindings, role_keys.bound, open_role_keys=role_keys.open),
            walks=role_class_walks,
            arcs_graph=graph,
            deadline=deadline,
        )
    except (RoleSignatureError, FamilyNotResolvableError):
        return None
    elected = resolution.value
    canonical = canonical_pin_bindings(elected, bindings, role_keys)
    return _Alternate(elected, canonical, not resolution.authority.settled, resolution.walked)


def settle_alternate(
    shape: ShapeDraft,
    role_keys: RoleKeys,
    rep_sv: str,
    *,
    scope: SourceScope | None = None,
    stray_values: Mapping[str, frozenset[str]] | None = None,
    bindings: Sequence[SlotBindingDraft] | None = None,
) -> _Alternate | None:
    """The alternate `walk_alternate` walks, before its walk: `shape` naming the dataset it
    is served from, with the constraint bindings of `rep_sv`, or the `bindings` a breakdown
    of the elected shape passes, in canonical form under `role_keys`. `None` where
    `walk_alternate` drops it because no table serves those bindings. Binding repair settles the elected shape's bindings this way before its role
    class is elected, and an alternate's whole total is judged on these, as the elected
    shape's is (`phase_place_materialise._walk_assignment`)."""
    served = _served_alternate(
        shape, role_keys, rep_sv, scope=scope, stray_values=stray_values or {}, bindings=bindings
    )
    if served is None:
        return None
    shape, bindings = served
    return _Alternate(shape, canonical_pin_bindings(shape, bindings, role_keys))


def _served_alternate(
    shape: ShapeDraft,
    role_keys: RoleKeys,
    rep_sv: str,
    *,
    scope: SourceScope | None,
    stray_values: Mapping[str, frozenset[str]],
    bindings: Sequence[SlotBindingDraft] | None,
) -> tuple[ShapeDraft, list[SlotBindingDraft]] | None:
    """`shape` naming the dataset the part is served from under the alternate's bindings
    (`part_source`), with those bindings naming it, or `None` when no table serves them
    (`bound_values.PartNotServed`). The bindings are `bindings` where given, else those of
    `rep_sv` with the drafted `stray_values` its members apply (`_with_stray_values`)."""
    if bindings is None:
        bindings = _with_stray_values(
            standard_bindings_from_arcs(shape=shape, rep_sv_dcid=rep_sv), shape, stray_values
        )
    else:
        bindings = list(bindings)
    try:
        source = part_source(shape, bindings, lambda: role_keys, scope=scope)
    except PartNotServed:
        return None
    if source is not None and shape.named_dataset is None:
        shape = shape_for_part(shape, named_scope(source))
    return shape, with_named_source(bindings, shape)


def elect_alternate(
    shape: ShapeDraft,
    role_keys: RoleKeys,
    *,
    elected_bindings: list[SlotBindingDraft],
    elected_has_data: bool,
    graph: EngineGraphClient,
    deadline: float | None = None,
    role_class_walks: RoleClassWalks | None = None,
) -> _Alternate | None:
    """`walk_alternate` at the alternate's representative (`alternate_representative`), or
    `None` when it has none."""
    rep_sv = alternate_representative(
        shape, elected_bindings=elected_bindings, elected_has_data=elected_has_data
    )
    if rep_sv is None:
        return None
    return walk_alternate(
        shape,
        role_keys,
        rep_sv,
        graph=graph,
        deadline=deadline,
        role_class_walks=role_class_walks,
    )


class _AlternatesRead(NamedTuple):
    """What `_materialise_alt_shapes` read: the candidates with data, an `UnsupportedDraft`
    that blocks candidate collapse, the presence read over (`presence` as given, topped up
    for alternates whose bindings unbinding relaxed), and each alternate left unevaluated
    because that top-up did not complete, with the reason."""

    candidates: list[_Candidate]
    blocking: UnsupportedDraft | None
    presence: ConsultedPresence | None
    unevaluated: tuple[tuple[_Alternate, AlternateUnevaluatedReason], ...]
    # Each alternate's own record of the scopes its reading consulted and the reads it lost
    # (`degraded_reads.carry`), in `alternates` order. The caller carries those of the
    # readings the answer stands on.
    consulted: tuple[tuple[_Alternate, DegradedReads], ...] = ()


def _materialise_alt_shapes(
    alternates: Sequence[_Alternate],
    *,
    places: Sequence[PlaceContext | UnsettledPlace],
    presence: ConsultedPresence | None = None,
    named_place_dcids: Sequence[str],
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    date_request: DateRequest | None,
    deadline: float | None = None,
    coverage: CoverageSnapshot,
    over_bound: list[_Alternate] | None = None,
    presence_failure: AlternateUnevaluatedReason | None = None,
) -> _AlternatesRead:
    """Materialise alternates (`elect_alternate`), each in its own place context.

    ``places`` holds each alternate's places, in ``alternates`` order, for the shape the
    alternate holds. An alternate whose place sets are unsettled settles them on its own
    read (`place_context.settle_sets`): its where slot, role entities and probe places come
    from the context it settles to, and each result carries that context settled on it. An
    alternate whose places exceed the bound on the places a Spec binds
    (`PlaceSetTooLargeError`) is dropped and appended to ``over_bound`` when one is given,
    so the caller can tell it from an alternate read and found empty. A context with no probe
    places has its cache warm read every named place (``named_place_dcids``), so no one
    place decides it. ``presence`` is the request's presence read, which a default-scoped
    shape's materialisation derives its scope from (`PlaceContext.presence_for`). Shapes
    without data are dropped, and so is one whose member the mint refuses because only a
    dataset the code index does not record for it publishes its observations
    (`DatasetDisagreementError`). An ``UnsupportedDraft`` blocks candidate collapse because
    the alternate is unevaluated. Values that match no member together are unbound and the
    alternate read again over ``presence`` topped up for its new bindings
    (`_presence_after_unbinding`); an alternate whose top-up does not complete, or that needs
    one after the pass's optional presence read failed (``presence_failure``) or after an
    earlier alternate's top-up in this call did, is unevaluated. The result carries the presence read over, which the caller merges into
    the request's (`_AlternatesRead`).

    Batched observation-facet requests warm the cache before the loop, avoiding up to
    ``max_candidates - 1`` sequential reads. A role-bearing shape resolves from coverage
    and is not warmed. Shapes probed over the same places share one
    batch, and every default-scoped shape joins one batch that reads each member at its own
    scope (`_prewarm_places`, `facets_at_own_places`), in first-shape order. A shape whose
    read decides a place set is warmed at every place either reading binds
    (`UnsettledPlace.read_places`). Each batch reads the members its materialisation reads,
    those of the family the alternate's own query elects (its bindings and role keys,
    `shape_query`), so its resolve hits the cache keyed by StatVar, entity, and date need.
    Each batch is capped at ``region_max_members``: index
    populations may span the full CRS vocabulary, while a recall-sourced shape holds
    only the members recall found. Materialisation is uncapped.
    """
    # Prewarm key: one for every default-scoped shape, else the probe places themselves.
    # Each batch maps its members to the places they are read at, the first shape's first.
    readable: list[tuple[_Alternate, PlaceContext | UnsettledPlace]] = []
    # What each alternate's own reading consulted and lost, in a record of its own.
    consulted: dict[int, DegradedReads] = {}
    prewarms: dict[tuple[str, ...] | None, dict[str, tuple[str, ...]]] = {}
    for alternate, unsettled in zip(alternates, places, strict=True):
        shape = alternate.shape
        record = consulted[id(alternate)] = DegradedReads()
        try:
            with recording_degraded_reads(detached=True, into=record):
                place = unread_context(unsettled)
        except PlaceSetTooLargeError as exc:
            _drop_over_bound(alternate, exc, over_bound)
            continue
        except PlaceSetNotAdmittedError:
            logger.info(
                "shape %r admits no reading of a place set; dropped from the offer",
                alternate.shape.shape_id,
            )
            continue
        readable.append((alternate, unsettled))
        if is_role_bearing(shape):
            # It resolves from coverage (`graph_confirm_resolve`), so it reads no warmed facet.
            continue
        role_keys = entity_role_keys(place.roles.resolved_entities, open_axes=place.roles.open_axes)
        shape_members, _index_sourced = shape_member_facts(
            shape,
            shape_query(shape, alternate.bindings, role_keys.bound, open_role_keys=role_keys.open),
        )
        places_of = _prewarm_places(
            shape,
            alternate.bindings,
            role_keys,
            list(shape_members),
            place,
            presence,
            named_place_dcids,
            unsettled.read_places if isinstance(unsettled, UnsettledPlace) else (),
        )
        if not any(places_of.values()):
            continue
        key = None if place.default_scope is not None else next(iter(places_of.values()))
        batch = prewarms.setdefault(key, {})
        for sv, sv_places in places_of.items():
            batch.setdefault(sv, sv_places)
    for batch in prewarms.values():
        members = list(batch)[: settings().region_max_members]
        facets_at_own_places(
            graph, {sv: batch[sv] for sv in members}, needs_dates=(date_request is not None)
        )

    results: list[_Candidate] = []
    unevaluated: list[tuple[_Alternate, AlternateUnevaluatedReason]] = []
    for alternate, unsettled in readable:
        with recording_degraded_reads(detached=True, into=consulted[id(alternate)]):
            bindings = list(alternate.bindings)
            # The presence the alternate is read over, the last topped up after unbinding.
            presences = [presence]

            def read(
                place: PlaceContext,
                facet_batch: FacetBatch | None,
                alternate: _Alternate = alternate,
                bindings: list[SlotBindingDraft] = bindings,
                presences: list[ConsultedPresence | None] = presences,
            ) -> Materialised | NoDataDraft | MaterialisedCandidates | UnsupportedDraft:
                return materialise(
                    alternate.shape,
                    with_where(bindings, place),
                    place.roles.resolved_entities,
                    graph=graph,
                    sdmx=sdmx,
                    date_request=date_request,
                    deadline=deadline,
                    place_dcids=place.probe_places,
                    default_presence=place.presence_for(presences[-1]),
                    discover_scope=True,
                    open_axes=place.roles.open_axes,
                    coverage=coverage,
                    facet_batch=facet_batch,
                )

            try:
                # The alternate's values come from its representative StatVar, none from the
                # request, so values that match no member together are unbound silently, a
                # minimal set per round (`BoundValuesUnrealisable.relaxation`).
                while True:
                    place, result = settle_sets(unsettled, read)
                    if not isinstance(result, BoundValuesUnrealisable):
                        break
                    bindings[:] = without_values(bindings, result.relaxation())
                    scoped = place.presence_for(presences[-1])
                    if scoped is not None:
                        covering = _presence_after_unbinding(
                            scoped,
                            alternate.shape,
                            bindings,
                            place,
                            graph=graph,
                            sdmx=sdmx,
                            deadline=deadline,
                            failure=presence_failure,
                        )
                        if not isinstance(covering, ConsultedPresence):
                            unevaluated.append((alternate, covering))
                            # The call tries no other optional presence read after one failed.
                            presence_failure = presence_failure or covering
                            break
                        # Later alternates read over what this one added.
                        presence = covering
                        presences.append(covering)
            except PlaceSetTooLargeError as exc:
                _drop_over_bound(alternate, exc, over_bound)
                continue
            except PlaceSetNotAdmittedError:
                logger.info(
                    "shape %r admits no reading of a place set; dropped from the offer",
                    alternate.shape.shape_id,
                )
                continue
            except DatasetDisagreementError:
                # The mint refuses this alternate's identity, so it is dropped as one whose walk
                # the mint refuses is; the elected answer stands.
                logger.warning(
                    "alternate %r dropped: a member's dataset disagrees with the code index",
                    alternate.shape.shape_id,
                    exc_info=True,
                )
                continue
            if isinstance(result, BoundValuesUnrealisable):
                # Its presence read under the relaxed bindings failed (`_presence_after_unbinding`).
                continue
            if isinstance(result, UnsupportedDraft):
                return _AlternatesRead(
                    results,
                    result,
                    presence,
                    tuple(unevaluated),
                    tuple((a, consulted[id(a)]) for a in alternates),
                )
            if isinstance(result, Materialised) and result.has_data:
                if alternate.unverified:
                    result = dataclasses.replace(result, membership_unverified=True)
                results.append(
                    _Candidate(
                        settled_shape(alternate.shape, result),
                        with_where(bindings, place),
                        result,
                        place.settled(result),
                        alternate.walked,
                    )
                )
    return _AlternatesRead(
        results,
        None,
        presence,
        tuple(unevaluated),
        tuple((a, consulted[id(a)]) for a in alternates),
    )


def _presence_after_unbinding(
    presence: ConsultedPresence,
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    place: PlaceContext,
    *,
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    deadline: float | None,
    failure: AlternateUnevaluatedReason | None,
) -> ConsultedPresence | AlternateUnevaluatedReason:
    """`presence` topped up with what the resolve of an alternate's `shape` under `bindings`,
    which unbinding relaxed, reads at its own default scope (`presence_covering`), or why
    that read did not complete, as `phase_place_materialise._await_alternates_presence`
    words it: the alternate is then unevaluated and the elected answer stands. After an
    optional presence read of the pass failed (`failure`), no other is tried, as
    `phase_place_materialise._read_alternates_presence` rules, and an alternate that needs
    one is unevaluated for that failure."""
    role_keys = entity_role_keys(place.roles.resolved_entities, open_axes=place.roles.open_axes)
    if failure is not None and presence_lacking(presence, shape, bindings, role_keys):
        return failure
    try:
        return presence_covering(
            presence, shape, bindings, role_keys, graph=graph, sdmx=sdmx, deadline=deadline
        )
    except (TimeoutError, GraphDeadlineExceededError):
        logger.warning("alternate %r unevaluated: its presence read timed out", shape.shape_id)
        return "timed_out"
    except (GraphInfraError, GraphReadError, GraphCapExceededError):
        logger.warning(
            "alternate %r unevaluated: its presence read failed", shape.shape_id, exc_info=True
        )
        return "upstream_error"


def _drop_over_bound(
    alternate: _Alternate, exc: PlaceSetTooLargeError, over_bound: list[_Alternate] | None
) -> None:
    """Log an alternate dropped for the bound on the places a Spec binds, and record it in
    `over_bound` when the caller keeps one."""
    logger.info(
        "shape %r binds %d places, over the bound of %d; dropped from the offer",
        alternate.shape.shape_id,
        exc.count,
        exc.bound,
    )
    if over_bound is not None:
        over_bound.append(alternate)


async def resolve_candidates(
    state: ResolveState, *, rate_cache: RateTableCache | None = None
) -> RegionResult | None:
    """Return a definite or candidates result when ambiguity yields data.

    Return ``None`` when a capped alternate blocks the path, when no alternate was
    evaluated, or when none survives beside an elected shape holding data only outside the
    years read.
    ``rate_cache`` is passed to bound-unit probes and single-survivor
    observations. Its default is a no-op.
    """
    from search.engine.region.candidate_admission import admit_alternates
    from search.engine.region.candidate_collapse import assemble_candidates

    admission = await admit_alternates(state)
    if admission.blocked or not admission.evaluated:
        return None
    return await assemble_candidates(state, admission, rate_cache=rate_cache)
