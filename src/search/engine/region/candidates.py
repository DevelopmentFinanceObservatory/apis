"""Build and collapse the ``also_plausible`` candidate fan-out.

Each candidate carries its own shape, bindings, materialised result and place context
(`_Candidate`). ``resolve_candidates`` returns ``None`` when no ambiguity survives.
Its sibling modules import helpers here, so it imports those modules locally
to avoid a cycle. Materialisation uses the shared retrieval interface.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from materialise.errors import DatasetDisagreementError
from materialise.place_scope import DefaultPresence
from materialise.rate_tables import RateTableCache
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.config import settings
from search.engine.errors import FamilyNotResolvableError, RoleSignatureError
from search.engine.extract import DateRequest
from search.engine.qualifier_match import _normalize
from search.engine.region.arcs import standard_bindings_from_arcs
from search.engine.region.codes import OBSERVATIONS_NOTHING_TO_PROBE
from search.engine.region.named_source import with_named_source
from search.engine.region.place_context import PlaceContext, settled_shape
from search.engine.region.place_roles import PlaceRoles, where_binding
from search.engine.region.result import RegionResult
from search.engine.region.state import ResolveState
from search.engine.retrieval.bound_values import (
    PartNotServed,
    RoleClassWalks,
    WalkedMembers,
    canonical_pin_bindings,
    elected_role_class_shape,
    entity_role_keys,
    part_source,
    shape_for_part,
    shape_member_facts,
    shape_query,
    without_values,
)
from search.engine.retrieval.representative import shape_agreement_representative
from search.engine.retrieval.retrieve import (
    BoundValuesUnrealisable,
    Materialised,
    UnsupportedDraft,
    materialise,
)
from search.engine.retrieval.standard import facets_at_own_places
from search.engine.shape import ShapeDraft, is_role_bearing
from search.models import ResolvedQualifier, SpecObservations, Warning, omitted_observations
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
    context it is grounded and read over (`PlaceContext.settled` on `mat`), and what the
    mint states of the members its walk settled (`MembershipResolution.walked`)."""

    shape: ShapeDraft
    bindings: list[SlotBindingDraft]
    mat: Materialised
    place: PlaceContext
    walked: WalkedMembers | None


def _prewarm_places(
    shape: ShapeDraft,
    members: Sequence[str],
    place: PlaceContext,
    presence: DefaultPresence | None,
    named_place_dcids: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """The places a shape's facet prewarm reads each of `members` at: on a default scope,
    each member of the presence read at its own scope (`DefaultPresence.scope`), else every
    member at the shape's probe places, else at every named place."""
    scoped = place.presence_for(presence)
    if scoped is not None:
        covered = shape.sv_arc_facts or {}
        return {sv: scoped.scope(sv).places for sv in members if sv in covered}
    entities = tuple(place.probe_dcids or named_place_dcids)
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


def alternate_representative(
    shape: ShapeDraft, *, elected_bindings: list[SlotBindingDraft], elected_has_data: bool
) -> str | None:
    """The StatVar an alternate derives its constraint bindings from, or `None` when it has
    none.

    The shape arrives narrowed to the members the alternate guard offers it through, those
    matching the elected reading first (`alternate_guard.offered_alternates`), so its first
    member reads as the elected quantity. With elected data, a standard alternate needs a
    representative matching the elected bound values or a bare member. A role-bearing shape
    uses its first member because its constraint axes describe flows, and so does any shape
    when the elected shape has no data."""
    if is_role_bearing(shape) or not elected_has_data:
        return next(iter(shape.sv_arc_facts or {}), None)
    elected_bound_values: dict[str, list[str]] = {
        b.property_dcid: list(b.value_dcids)
        for b in elected_bindings
        if b.property_dcid and b.kind in ("value", "set") and b.value_dcids
    }
    return shape_agreement_representative(shape, elected_bound_values)


def walk_alternate(
    shape: ShapeDraft,
    roles: PlaceRoles,
    rep_sv: str,
    *,
    graph: EngineGraphClient,
    deadline: float | None = None,
    role_class_walks: RoleClassWalks | None = None,
    excluded: frozenset[str] = frozenset(),
) -> _Alternate | None:
    """Give an alternate the constraint bindings of `rep_sv` and narrow it to the role class
    its entities elect, or return `None` when the mint refuses its identity or no table the
    part may answer the alternate from holds a row for those bindings
    (`bound_values.PartNotServed`), so an alternate is never served from a table the part
    excludes or from a table of another family.

    The chosen shape alone receives model bindings. An alternate derives constraint bindings
    from its representative StatVar (`alternate_representative`), and names the dataset its
    shape names (`with_named_source`), as the chosen shape does, so its identity is the one
    a question electing it mints. When the shape names none, the alternate names the
    dataset the part, which excludes the datasets `excluded`, is served from under those
    bindings and its role keys, where its unnamed identity would resolve to another at the
    mint (`part_source`), as the elected shape is named under the model's bindings. Its
    bindings on the invariants
    of the dataset its family elects take their canonical form as the chosen shape's do
    (`canonical_pin_bindings`), so its slots state what its canonical identity binds.

    ``roles`` is the alternate's binding of the query's places. Its entities, keyed as the
    alternate's own identity binds them, elect one role class among its members
    (``elected_role_class_shape``), settled by the mint's walk when ``role_class_walks`` is
    given, whether or not the alternate declares a flow role, as the elected shape's is. An
    alternate whose members the walk does not settle is marked `unverified`, and its result
    carries ``membership_unverified``. A walk that meets role sets the election cannot
    order, or a member at fault, drops the alternate, as the mint refuses its identity. The
    walk can change the roles the shape declares, so the caller binds the returned shape's
    places (`place_roles.bound_after_election`) and builds its place context after this
    call.
    """
    bindings = standard_bindings_from_arcs(shape=shape, rep_sv_dcid=rep_sv)
    role_keys = entity_role_keys(roles.resolved_entities, open_axes=roles.open_axes)
    try:
        source = part_source(shape, bindings, lambda: role_keys, excluded=excluded)
    except PartNotServed:
        return None
    if source is not None and shape.named_dataset is None:
        shape = shape_for_part(shape, source)
    bindings = with_named_source(bindings, shape)
    where = where_binding(roles.where_dcids)
    try:
        resolution = elected_role_class_shape(
            shape,
            shape_query(
                shape,
                bindings if where is None else [*bindings, where],
                role_keys.bound,
                open_role_keys=role_keys.open,
            ),
            walks=role_class_walks,
            arcs_graph=graph,
            deadline=deadline,
        )
    except (RoleSignatureError, FamilyNotResolvableError):
        return None
    elected = resolution.value
    canonical = canonical_pin_bindings(elected, bindings, role_keys)
    return _Alternate(elected, canonical, not resolution.authority.settled, resolution.walked)


def elect_alternate(
    shape: ShapeDraft,
    roles: PlaceRoles,
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
        roles,
        rep_sv,
        graph=graph,
        deadline=deadline,
        role_class_walks=role_class_walks,
    )


def _materialise_alt_shapes(
    alternates: Sequence[_Alternate],
    *,
    places: Sequence[PlaceContext],
    presence: DefaultPresence | None = None,
    named_place_dcids: Sequence[str],
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    date_request: DateRequest | None,
    deadline: float | None = None,
    coverage: CoverageSnapshot,
) -> tuple[list[_Candidate], UnsupportedDraft | None]:
    """Materialise alternates (`elect_alternate`), each in its own place context.

    ``places`` holds each alternate's place context, in ``alternates`` order, built for
    the shape the alternate holds: its where slot, role entities and probe places come from
    there, and each result carries the context settled on it. A context with no probe
    places has its cache warm read every named place (``named_place_dcids``), so no one
    place decides it. ``presence`` is the request's presence read, which a default-scoped
    shape's materialisation derives its scope from (`PlaceContext.presence_for`). Shapes
    without data are dropped, and so is one whose member the mint refuses because only a
    dataset the code index does not record for it publishes its observations
    (`DatasetDisagreementError`). An ``UnsupportedDraft`` blocks candidate collapse because
    the alternate is unevaluated.

    Batched observation-facet requests warm the cache before the loop, avoiding up to
    ``max_candidates - 1`` sequential reads. Shapes probed over the same places share one
    batch, and every default-scoped shape joins one batch that reads each member at its own
    scope (`_prewarm_places`, `facets_at_own_places`), in first-shape order. Each batch reads
    the members its materialisation reads, those of the family the alternate's own query
    elects (its bindings and role keys, `shape_query`), so its resolve hits the cache keyed
    by StatVar, entity, and date need. Each batch is capped at ``region_max_members``: index
    populations may span the full CRS vocabulary, while recall is capped at
    ``SEARCH_MAX_CONFIRM_CANDIDATES`` per shape. Materialisation is uncapped.
    """
    # Prewarm key: one for every default-scoped shape, else the probe places themselves.
    # Each batch maps its members to the places they are read at, the first shape's first.
    prewarms: dict[tuple[str, ...] | None, dict[str, tuple[str, ...]]] = {}
    for alternate, place in zip(alternates, places, strict=True):
        shape = alternate.shape
        role_keys = entity_role_keys(place.roles.resolved_entities, open_axes=place.roles.open_axes)
        shape_members, _index_sourced = shape_member_facts(
            shape,
            shape_query(shape, alternate.bindings, role_keys.bound, open_role_keys=role_keys.open),
        )
        places_of = _prewarm_places(shape, list(shape_members), place, presence, named_place_dcids)
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
    for alternate, place in zip(alternates, places, strict=True):
        bindings = list(alternate.bindings)
        if place.where is not None:
            bindings.append(place.where)
        try:
            # The alternate's values come from its representative StatVar, none from the
            # request, so values that match no member together are unbound silently, a
            # minimal set per round (`BoundValuesUnrealisable.relaxation`).
            while True:
                result = materialise(
                    alternate.shape,
                    bindings,
                    place.roles.resolved_entities,
                    graph=graph,
                    sdmx=sdmx,
                    date_request=date_request,
                    deadline=deadline,
                    place_dcids=place.probe_places,
                    default_presence=place.presence_for(presence),
                    discover_scope=True,
                    open_axes=place.roles.open_axes,
                    coverage=coverage,
                )
                if not isinstance(result, BoundValuesUnrealisable):
                    break
                bindings = without_values(bindings, result.relaxation())
        except DatasetDisagreementError:
            # The mint refuses this alternate's identity, so it is dropped as one whose walk
            # the mint refuses is; the elected answer stands.
            logger.warning(
                "alternate %r dropped: a member's dataset disagrees with the code index",
                alternate.shape.shape_id,
                exc_info=True,
            )
            continue
        if isinstance(result, UnsupportedDraft):
            return results, result
        if isinstance(result, Materialised) and result.has_data:
            if alternate.unverified:
                result = dataclasses.replace(result, membership_unverified=True)
            results.append(
                _Candidate(
                    settled_shape(alternate.shape, result),
                    bindings,
                    result,
                    place.settled(result),
                    alternate.walked,
                )
            )
    return results, None


def _unbound_query_qualifiers(
    query_qualifiers: Sequence[ResolvedQualifier], *, unapplied: frozenset[str]
) -> tuple[ResolvedQualifier, ...]:
    """Return unbound and unsupported qualifiers from the elected shape's query.

    This includes unsupported phrases. A term whose normalised text is in `unapplied` is
    left out. Its values matched no member together with the other bound values, so the
    answer reads without them and already departs it (`unrealisable.unbind_unrealisable`).
    The pipeline and single-survivor collapse share the predicate. It lives here to avoid
    an import cycle.
    """
    return tuple(
        q
        for q in query_qualifiers
        if q.disposition != "bound" and _normalize(q.text) not in unapplied
    )


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
