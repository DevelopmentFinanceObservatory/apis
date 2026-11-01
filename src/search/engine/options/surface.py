"""Build the options surface for a resolved identity population.

`observationAbout` enumerates places and keeps candidates with data for any
population member. `origin` and `destination` group coverage by axis and
filter on entities bound to the opposite axis. Resolution and role validation
happen in `options.resolve`; this module builds candidates and the response
envelope.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from code_index.shared import get_index
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from dfo_platform.build import build_commit
from entity_resolution import shared_membership_policy
from materialise import GraphReader
from materialise.enrichment import read_party_kinds
from materialise.errors import DefaultFacetUndefinedError, FacetNotAvailableError
from materialise.recipes import (
    NonCurrencyUnitError,
    RateCoverageGapError,
    RecipeNotDeclaredError,
    RoleAxisConflictError,
    RoleAxisNotDeclaredError,
    cause_from_recipe_error,
)
from origin_catalogue import current_catalogue, recording_un_blocked_places
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.materialise_bridge import observation_facets_via_materialise
from search.engine.config import settings
from search.engine.envelope import make_diagnostics, now_ms
from search.engine.errors import (
    EngineInfraError,
    EngineInputError,
    GraphCapExceededError,
    GraphInfraError,
)
from search.engine.extract import DateRequest
from search.engine.facet_refusals import (
    RELEVANT_FACET_AXES,
    facet_satisfies_bindings,
    facets_satisfy_bindings,
)
from search.engine.place import REGION_MEMBER_TYPE, expand_places
from search.engine.region import OPTION_COVERAGE_PROBE_FAILED
from search.engine.region.convertibility import (
    BoundFacetCombinationsCeilingExceededError,
    bound_facet_refs,
    decide_option_units,
    elected_unit_refs,
)
from search.engine.region.resubmit_request import (
    _source_dcids_from_facets_provenance,
    pinned_provenance_scope,
)
from search.engine.region.un_system import NO_LABELS, un_unavailable_warning
from search.engine.retrieval.coverage import CoverageForm, answer_coverage
from search.engine.retrieval.elected_dataset import published
from search.engine.retrieval.read_plan import plan_standard, plan_years, reads_dates
from search.models import (
    GraphRef,
    IdentityFacetBinding,
    IdentityFacetValueRequest,
    OptionsRequest,
    OptionsResponse,
    OptionsSet,
    RequestedWindow,
    ResolvedEntity,
    RoleAxis,
    SlotOption,
    SlotValue,
    TimeWindow,
    Warning,
)
from series_coverage.shared import CoverageSnapshot

from .resolve import ResolvedOptionsTarget

if TYPE_CHECKING:
    from series_coverage.index import BoundRoles
    from series_coverage.index import Role as CoverageRole

logger = logging.getLogger(__name__)

# The flow axis on the opposite end from a given one, for filtering a
# directional availability read by whichever end the spec already binds.
_OTHER_FLOW_AXIS: dict[RoleAxis, RoleAxis] = {"origin": "destination", "destination": "origin"}

# ---------------------------------------------------------------------------
# observationAbout: anchor-type routing tables -- see
# _PEER_TYPES/_CHILD_TYPES below.
# ---------------------------------------------------------------------------

# "peers" scope: the anchor's own typeOf, enumerated under this root. A
# sub-country anchor type (e.g. AdministrativeArea1) has no single global root
# its true peers sit under, so only the two rooted-at-Earth types are
# supported; any other anchor type is a typed refusal
# (unsupported_peers_anchor_type), never a silently-empty result.
_PEERS_ROOT = "Earth"
_PEER_TYPES: frozenset[str] = frozenset({"Country", "Continent"})

# The deployment's most common place type: the peers-scope default when no
# observationAbout entity is bound at all (a legal, likely-common request --
# "nothing picked yet, which places have data?"). "children" has no analogous
# default -- see _observation_about_options_set's no-anchor branch.
_DEFAULT_PEER_TYPE = "Country"

# "children" scope: anchor typeOf -> the child typeOf to enumerate under the
# anchor itself. One level deep per anchor type. An anchor type absent from
# this table is a typed refusal (unsupported_children_anchor_type); no anchor
# bound at all is a different typed refusal (children_anchor_required) --
# see the scope == "children" branch below for why the two are kept distinct.
_CHILD_TYPES: dict[str, str] = {
    "Planet": "Continent",
    "Continent": "Country",
    "Country": "AdministrativeArea1",
}


def _date_request_from_window(window: RequestedWindow | None) -> DateRequest | None:
    """The target's own requested_window, re-expressed as the DateRequest
    option coverage plans with. Mirrors region/resubmit_request.py's
    private _date_request_for_resubmit, which this module cannot reuse
    directly (it reads a resubmit identity). The options door derives its
    window from the posted identity in options/resolve.py, which folds the
    requested-window/latest sentinel down to one value before the surface
    half sees it.
    """
    if window is None:
        return None
    if isinstance(window, str):  # the "latest" sentinel
        return DateRequest(window=None, latest=True)
    if isinstance(window, TimeWindow):
        return DateRequest(window=window, latest=False)
    raise ValueError(f"unknown RequestedWindow kind: {window!r}")  # pragma: no cover


def _entities_with_role(entities: list[ResolvedEntity], role: str) -> list[ResolvedEntity]:
    return [e for e in entities if e.role == role]


# ---------------------------------------------------------------------------
# observationAbout (standard): graph enumeration intersected against data
# ---------------------------------------------------------------------------


def _resolve_scope_target(req: OptionsRequest, target: ResolvedOptionsTarget) -> tuple[str, str]:
    """Resolve an observationAbout options request's own (parent_dcid,
    child_type) from req.for_.scope ("peers" or "children") and the
    request's bound observationAbout anchors.

    "peers" enumerates the anchor's own typeOf under Earth (or Country, the
    deployment's most common type, when no anchor is bound at all); "peers"
    is computed over the SET of resolved anchor types so several anchors
    sharing one type yield the identical candidate universe regardless of
    posting order. "children" descends one level from exactly one bound
    anchor via _CHILD_TYPES; unlike "peers" there is no anchor-free default,
    since computing a place's children requires already knowing the place. A
    continent or subregion by the membership policy
    (`MembershipPolicy.holds_region`) has countries as children whatever
    its graph type.

    Raises EngineInputError for every anchor shape either scope does not
    support: mixed anchor types under "peers" (mixed_peers_anchor_types), an
    anchor type this door has no peers/children mapping for
    (unsupported_peers_anchor_type / unsupported_children_anchor_type), a
    "children" request with no bound anchor (children_anchor_required), or
    more than one distinct anchor dcid under "children"
    (children_anchor_ambiguous).
    """
    anchors = _entities_with_role(list(target.anchors), "observationAbout")
    scope = req.for_.scope or "peers"

    if scope == "peers":
        if not anchors:
            return _PEERS_ROOT, _DEFAULT_PEER_TYPE
        anchor_types = {a.entity_type for a in anchors}
        if len(anchor_types) > 1:
            raise EngineInputError(
                "peers scope requires every bound observationAbout anchor to "
                "share one entity type; the bound anchors have types "
                f"{', '.join(sorted(t or 'unknown' for t in anchor_types))}",
                code="mixed_peers_anchor_types",
                pointer="/identity/entities/observationAbout",
            )
        (child_type,) = anchor_types
        if child_type not in _PEER_TYPES:
            raise EngineInputError(
                f"peers scope not supported for anchor type {child_type!r}",
                code="unsupported_peers_anchor_type",
                pointer="/for/scope",
            )
        return _PEERS_ROOT, child_type

    # scope == "children"
    if not anchors:
        raise EngineInputError(
            "children scope requires a bound observationAbout entity",
            code="children_anchor_required",
            pointer="/for/scope",
        )
    # Computed over the SET of anchor dcids (order- and duplicate-invariant,
    # matching the identity core's own bucket normalization) rather than the
    # posted list, so a client cannot dodge the refusal by posting the same
    # dcid twice.
    anchor_dcids = {a.ref for a in anchors}
    if len(anchor_dcids) > 1:
        raise EngineInputError(
            "children scope requires exactly one bound observationAbout anchor; "
            f"{len(anchor_dcids)} distinct anchors are bound",
            code="children_anchor_ambiguous",
            pointer="/identity/entities/observationAbout",
        )
    anchor = anchors[0]
    if shared_membership_policy().holds_region(anchor.ref):
        return anchor.ref, REGION_MEMBER_TYPE
    resolved_child_type = _CHILD_TYPES.get(anchor.entity_type or "")
    if resolved_child_type is None:
        raise EngineInputError(
            f"children scope not supported for anchor type {anchor.entity_type!r}",
            code="unsupported_children_anchor_type",
            pointer="/for/scope",
        )
    return anchor.ref, resolved_child_type


def _fetch_candidates(
    graph: EngineGraphClient, parent_dcid: str, child_type: str
) -> tuple[list[str], bool]:
    """Return (candidates, candidates_truncated), capped to settings().max_options.

    A region's candidates are the countries the membership policy holds for it, sorted
    (`place.expand_places`), the members an answer binds for it; a region the policy omits
    has none. The policy decides what is a region (`MembershipPolicy.holds_region`), whatever
    the node's graph type. Any other parent's are its own children of child_type via
    graph.child_places_batch, the World among them.
    """
    membership = shared_membership_policy()
    if membership.holds_region(parent_dcid):
        expansion = expand_places([parent_dcid], membership=membership)
        found = expansion[parent_dcid]
        candidates = [] if found is None else found.children
    else:
        candidates = graph.child_places_batch([parent_dcid], child_type).get(parent_dcid, [])
    candidates_truncated = len(candidates) > settings().max_options
    if candidates_truncated:
        candidates = candidates[: settings().max_options]
    return candidates, candidates_truncated


def _presence_filter(
    graph: EngineGraphClient, sv_dcids: list[str], candidates: list[str]
) -> list[str]:
    """Filter candidates to those graph.entities_with_data confirms hold data
    for AT LEAST ONE of sv_dcids (union presence, not intersection -- see
    _observation_about_options_set's own docstring). Raises EngineInputError
    (scope_too_broad) when the presence check's own max_bytes cap trips.
    """
    try:
        return graph.entities_with_data(sv_dcids, candidates, max_bytes=settings().sdmx_max_bytes)
    except GraphCapExceededError as exc:
        raise EngineInputError(
            f"the candidate presence check against {len(candidates)} candidates "
            f"exceeded its {exc.cap}-{exc.cap_kind} response cap ({exc.observed} "
            "observed); bind an additional slot to narrow the population",
            code="scope_too_broad",
            status=422,
            pointer="/for/role",
        ) from exc


def _decide_elected_units(
    facet_bindings: dict[str, IdentityFacetBinding],
    sv_dcids: list[str],
    present: list[str],
    target: ResolvedOptionsTarget,
    *,
    materialise_graph: GraphReader | None,
    materialise_sdmx: MaterialiseSdmxClient | None,
    coverage_snapshot: CoverageSnapshot,
    deadline: float | None,
) -> Mapping[str, frozenset[str]] | None:
    """Decide a request-form unit binding's own convertibility for each
    stat_var alone (region/convertibility.py's decide_option_units): the
    options are alternatives, not members of one Spec, so each elects as a
    Spec of that one stat_var would. A facet is admitted only when its unit is one a mint
    would elect for that stat_var, given every candidate still present
    after the presence filter. Decided once per stat_var, over every
    present candidate: elected_units is a per-stat_var property, and this
    door has no single bound observationAbout entity to decide it against.
    declared_roles is fixed at ("observationAbout",) because this function
    enumerates a subject-only family's candidates. None when no unit
    binding is in play.

    Raises EngineInfraError when materialise_graph/materialise_sdmx are not
    initialised, or EngineInputError for every recipe/coverage/ceiling
    failure decide_option_units itself raises.
    """
    unit_binding = facet_bindings.get("unit")
    if not isinstance(unit_binding, IdentityFacetValueRequest):
        return None
    if materialise_graph is None or materialise_sdmx is None:
        raise EngineInfraError(
            "materialise graph/sdmx client not initialised; cannot decide this "
            "bound unit request's own convertibility"
        )
    bound_facets = bound_facet_refs(
        facet_bindings, _source_dcids_from_facets_provenance(facet_bindings)
    )
    try:
        elected = decide_option_units(
            sv_dcids,
            unit_request=unit_binding.request,
            role_bindings={"observationAbout": present},
            declared_roles=("observationAbout",),
            bound_facets=bound_facets,
            # The years this door's option coverage plans with.
            years=plan_years(_date_request_from_window(target.requested_window)),
            graph=materialise_graph,
            sdmx=materialise_sdmx,
            coverage=coverage_snapshot.index,
            datasets=target.population.datasets,
            deadline=deadline,
        )
    except NonCurrencyUnitError as exc:
        detail = str(exc)
        raise EngineInputError(
            detail,
            code=exc.code,
            status=exc.status,
            pointer="/identity/facets/unit",
        ) from exc
    except (RoleAxisNotDeclaredError, RoleAxisConflictError) as exc:
        # Both mean "this axis cannot give me one role" -- the axis is
        # undeclared, or two declared roles share it. The same
        # request_invalid 422 with the same pointer fits both.
        detail = str(exc)
        raise EngineInputError(
            detail,
            code="request_invalid",
            status=422,
            pointer="/identity/facets/unit",
        ) from exc
    except BoundFacetCombinationsCeilingExceededError as exc:
        # The Set-bound provenance/measurementMethod product would exceed
        # _BOUND_FACET_COMBINATIONS_MAX -- refused before a single
        # decide_member_unit probe runs, not after.
        detail = str(exc)
        raise EngineInputError(
            detail,
            code="facet_probe_ceiling",
            status=422,
            pointer="/identity/facets/unit",
        ) from exc
    except (RecipeNotDeclaredError, RateCoverageGapError) as exc:
        detail = exc.detail or str(exc)
        raise EngineInputError(
            detail,
            code=exc.code,
            status=exc.status,
            pointer="/identity/facets/unit",
            cause=cause_from_recipe_error(exc),
        ) from exc
    except (FacetNotAvailableError, DefaultFacetUndefinedError) as exc:
        detail = exc.detail or str(exc)
        raise EngineInputError(
            detail,
            code="facet_not_available",
            status=422,
            pointer="/identity/facets/unit",
        ) from exc
    return elected_unit_refs(elected)


def _fetch_facets_batch(
    facet_bindings: dict[str, IdentityFacetBinding],
    *,
    include_coverage: bool,
    sv_dcids: list[str],
    present: list[str],
    materialise_graph: GraphReader | None,
    needs_dates: bool,
    deadline: float | None,
) -> tuple[dict | None, Warning | None]:
    """Fetch every present candidate's own observation facets, with their dates under
    `needs_dates`, when a facet binding or include_coverage needs them. Only coverage reads
    dates; membership judges facets by their axes alone. Neither fetch carries a
    StatVar-count precheck of its own (observation_facets_via_materialise is
    already chunked and process-wide concurrency-limited, and deadline
    bounds it against the door's own remaining request budget).

    A relevant facet binding makes this fetch load-bearing for MEMBERSHIP: a
    candidate belongs in the option set only because a satisfying facet was
    found, so every exception -- GraphInfraError or otherwise -- propagates
    instead of degrading to a presence-only answer (GraphInfraError reaches
    app.py's classify_search_exception as a retryable 503). With no facet
    binding in play, the same fetch runs only to satisfy include_coverage's
    optional enrichment and fails open on a GraphInfraError, returning an
    OPTION_COVERAGE_PROBE_FAILED warning instead of raising -- present/labels
    are already computed by this point, so a transport failure must not
    cost the caller the candidates it already has.
    """
    if facet_bindings:
        if materialise_graph is None:
            raise EngineInfraError(
                "materialise graph client not initialised; cannot confirm "
                "facet-bound option membership"
            )
        facets_batch = observation_facets_via_materialise(
            stat_vars=sv_dcids,
            entities=present,
            materialise_graph=materialise_graph,
            needs_dates=needs_dates,
            deadline=deadline,
        )
        return facets_batch, None

    if include_coverage and materialise_graph is not None:
        try:
            facets_batch = observation_facets_via_materialise(
                stat_vars=sv_dcids,
                entities=present,
                materialise_graph=materialise_graph,
                needs_dates=needs_dates,
                deadline=deadline,
            )
        except GraphInfraError:
            logger.warning(
                "observation_facets_via_materialise coverage probe failed for the "
                "options door's observationAbout role; returning uncovered options",
                exc_info=True,
            )
            warning = Warning(
                code=OPTION_COVERAGE_PROBE_FAILED,
                severity="warn",
                message=("observation coverage probe failed; options returned without coverage."),
            )
            return None, warning
        return facets_batch, None

    return None, None


def _filter_present_by_facets(
    graph: EngineGraphClient,
    present: list[str],
    sv_dcids: list[str],
    facet_bindings: dict[str, IdentityFacetBinding],
    facets_batch: dict | None,
    elected_units: Mapping[str, frozenset[str]] | None,
    *,
    deadline: float | None,
) -> list[str]:
    """Filter present to the candidates where AT LEAST ONE resolved
    StatVar's OWN facet list satisfies every bound facet axis under
    facet_refusal's all-or-nothing rule (a set binding's members are not
    alternatives). any(), not all(), across sv_dcids: membership itself is
    union presence, so a candidate a plain presence check already admits on
    one StatVar's data must not be excluded here just because every OTHER
    StatVar in a large population carries no facets for it at all. Checked
    per StatVar, never pooled across StatVars (see
    _observation_about_options_set's own docstring for why pooling here
    would be the wrong shape). present is returned unchanged when there is
    no relevant facet binding, or when facets_batch was not fetched.
    """
    if not facet_bindings or facets_batch is None:
        return present
    return [
        dcid
        for dcid in present
        if any(
            facets_satisfy_bindings(
                graph,
                facets_batch.get(sv, {}).get(dcid, []),
                facet_bindings,
                sv=sv,
                elected_units=elected_units,
                deadline=deadline,
            )
            for sv in sv_dcids
        )
    ]


def _build_observation_about_options(
    graph: EngineGraphClient,
    present: list[str],
    *,
    include_coverage: bool,
    facets_batch: dict | None,
    sv_dcids: list[str],
    facet_bindings: dict[str, IdentityFacetBinding],
    elected_units: Mapping[str, frozenset[str]] | None,
    date_request: DateRequest | None,
    deadline: float | None,
) -> list[SlotOption]:
    """Label every present candidate and attach its coverage when
    include_coverage asked for it, pooling facets across every resolved
    StatVar the same way membership's own facets_batch reads are (each
    facet stays tagged with its stat_var so elected_units can key the
    lookup correctly). A dcid that fails to resolve a label is omitted,
    never fabricated.
    """
    labels = graph.node_labels_batch(present)
    items: list[SlotOption] = []
    for dcid in present:
        label = labels.get(dcid)
        if label is None:
            continue
        coverage = None
        if include_coverage and facets_batch is not None:
            series = {
                (sv, dcid): [
                    f
                    for f in facets_batch.get(sv, {}).get(dcid, [])
                    if not facet_bindings
                    or facet_satisfies_bindings(
                        graph,
                        f,
                        facet_bindings,
                        sv=sv,
                        elected_units=elected_units,
                        deadline=deadline,
                    )
                ]
                for sv in sv_dcids
            }
            coverage = answer_coverage(
                plan_standard(
                    series,
                    members=sv_dcids,
                    years=plan_years(date_request),
                    places=(dcid,),
                    deadline=deadline,
                ),
                CoverageForm(facet_label="sources", obs_label="years"),
            )
        items.append(
            SlotOption(
                value=SlotValue(ref=GraphRef(ref=dcid, label=label), value_kind="entity"),
                coverage=coverage,
            )
        )
    return items


def _observation_about_options_set(
    req: OptionsRequest,
    target: ResolvedOptionsTarget,
    *,
    graph: EngineGraphClient,
    materialise_graph: GraphReader | None,
    materialise_sdmx: MaterialiseSdmxClient | None,
    include_coverage: bool,
    coverage_snapshot: CoverageSnapshot,
    deadline: float | None = None,
) -> tuple[OptionsSet, Warning | None]:
    """Enumerate places, keep those with data for any population member, and label them.

    Candidate order follows `_fetch_candidates`, then the order-preserving
    presence and facet filters; labels do not reorder results. A relevant
    facet binding requires at least one StatVar's own facet list to satisfy
    every bound axis. Coverage pools facets across StatVars and counts those
    satisfying any bound member. Facet reads required for membership propagate
    failures; coverage-only reads fail open with a warning.

    Request-form units use `decide_option_units` across all present candidates;
    only each StatVar's elected unit refs qualify. The result is truncated when its
    candidate list was capped.
    """
    parent_dcid, child_type = _resolve_scope_target(req, target)
    candidates, truncated = _fetch_candidates(graph, parent_dcid, child_type)

    sv_dcids = list(target.population.stat_var_dcids)
    if not candidates or not sv_dcids:
        return OptionsSet(items=[], truncated=truncated), None

    # Membership is presence at all, deliberately not presence inside the
    # requested window: entities_with_data takes no dates, and has_data is
    # computed over an entity's full history everywhere else in the API
    # (retrieval.coverage.answer_coverage), so filtering here on the
    # window would make an option mean something narrower on this door than
    # on every other surface. The window still reaches the caller, per
    # item, through `coverage` below, which is date-scoped.
    present = _presence_filter(graph, sv_dcids, candidates)
    if not present:
        return OptionsSet(items=[], truncated=truncated), None

    facet_bindings: dict[str, IdentityFacetBinding] = {
        axis: binding
        for axis, binding in (target.facets or {}).items()
        if axis in RELEVANT_FACET_AXES
    }
    elected_units = _decide_elected_units(
        facet_bindings,
        sv_dcids,
        present,
        target,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage_snapshot=coverage_snapshot,
        deadline=deadline,
    )

    # Under a range the coverage read carries each facet's dates (`reads_dates`), so a
    # facet with no point in the window leaves the plan, as it does in the answer, and
    # the option's coverage counts the points in the window. Under `latest` or every
    # date it reads none, since no election there judges a facet by its dates. A read for
    # membership alone reads none either: `facets_satisfy_bindings` reads no dates.
    date_request = _date_request_from_window(target.requested_window)
    facets_batch, warning = _fetch_facets_batch(
        facet_bindings,
        include_coverage=include_coverage,
        sv_dcids=sv_dcids,
        present=present,
        materialise_graph=materialise_graph,
        needs_dates=include_coverage and reads_dates(date_request),
        deadline=deadline,
    )
    if facets_batch is not None:
        # A member of an index-resolved family is judged on its elected dataset's facets.
        facets_batch = {
            sv: {
                entity: published(target.population.datasets, sv, facets)
                for entity, facets in by_entity.items()
            }
            for sv, by_entity in facets_batch.items()
        }

    present = _filter_present_by_facets(
        graph, present, sv_dcids, facet_bindings, facets_batch, elected_units, deadline=deadline
    )
    if not present:
        return OptionsSet(items=[], truncated=truncated), warning

    items = _build_observation_about_options(
        graph,
        present,
        include_coverage=include_coverage,
        facets_batch=facets_batch,
        sv_dcids=sv_dcids,
        facet_bindings=facet_bindings,
        elected_units=elected_units,
        date_request=date_request,
        deadline=deadline,
    )
    return OptionsSet(items=items, truncated=truncated), warning


# ---------------------------------------------------------------------------
# origin/destination: one coverage group-by on the axis's own role
# ---------------------------------------------------------------------------

# The fixed flow-axis -> series_coverage.index.Role pairing this module's own
# coverage lookup keys its `bound`/`axis` arguments by -- must agree with
# retrieval.role_scope's own _COVERAGE_ROLE_FOR_AXIS and retrieval
# .role_resolve.graph_confirm_resolve's identical mapping (series_coverage
# .schema's own module docstring is the source of truth both mirror).
_COVERAGE_ROLE_FOR_AXIS: dict[RoleAxis, CoverageRole] = {
    "destination": "recipient",
    "origin": "donor",
}


def _directional_opposite_bound(target: ResolvedOptionsTarget, *, axis: RoleAxis) -> BoundRoles:
    """The opposite flow axis's own bound entity, as a coverage BoundRoles
    map. It narrows a destination read by a bound origin and vice versa. {}
    (no binding) when the opposite axis carries no bound entity: there is
    nothing to narrow by. This needs no declared-role-name check: an axis
    this module never declared simply enumerates no anchors
    (_entities_with_role finds none), which already yields the same
    "nothing to narrow by" outcome.
    """
    other_axis = _OTHER_FLOW_AXIS[axis]
    other_dcids = [e.ref for e in _entities_with_role(list(target.anchors), other_axis)]
    if not other_dcids:
        return {}
    return {_COVERAGE_ROLE_FOR_AXIS[other_axis]: frozenset(other_dcids)}


def _directional_window_years(window: RequestedWindow | None) -> tuple[int | None, int | None]:
    """`window` re-expressed as the `first_year`/`last_year` pair
    `CoverageIndex.open_axis` takes: `None` for both when `window` is `None`
    (no date signal posted at all) or the "latest" sentinel (a latest ask is
    answered from whichever years the enumerated members turn out to carry,
    never narrowed by time), otherwise the `TimeWindow`'s own bounds
    unchanged -- open on either side stays open. `open_axis`'s own
    `CoverageIndex._apply_window` clamps whatever range results to the
    artifact's representable years and returns no rows for a window entirely
    outside it, so this need not float or cap the bounds itself the way
    `retrieval.read_plan.role_years` does for a wire SDMX read -- a local
    table filter has no request-size cap to protect.
    """
    if window is None or isinstance(window, str):
        return None, None
    return window.start_year, window.end_year


def _directional_options_set(
    target: ResolvedOptionsTarget,
    *,
    graph: EngineGraphClient,
    axis: RoleAxis,
    coverage_snapshot: CoverageSnapshot,
) -> OptionsSet:
    """Group coverage by `axis`, filtered by opposite-axis entities and the requested window.

    Coverage already contains members with data, so no separate presence
    probe is needed. Sort members before labelling for deterministic output;
    omit dcids without labels. Each item carries the party kind the code index lists for it in
    the datasets of the population's StatVars (`materialise.enrichment.read_party_kinds`), none
    for a party whose kind differs between them. No network cap applies to this local lookup,
    so the set is never truncated.
    """
    sv_dcids = list(target.population.stat_var_dcids)
    coverage_index = coverage_snapshot.index
    first_year, last_year = _directional_window_years(target.requested_window)
    members = sorted(
        coverage_index.open_axis(
            variables=sv_dcids,
            axis=_COVERAGE_ROLE_FOR_AXIS[axis],
            bound=_directional_opposite_bound(target, axis=axis),
            first_year=first_year,
            last_year=last_year,
        )
    )
    if not members:
        return OptionsSet(items=[], truncated=False)

    labels = graph.node_labels_batch(members)
    index = get_index()
    kinds = read_party_kinds(
        ((axis, dcid) for dcid in members if dcid in labels),
        parties=index.parties,
        datasets=index.statvar_provenances(sv_dcids),
    )
    items = [
        SlotOption(
            value=SlotValue(ref=GraphRef(ref=dcid, label=label), value_kind="entity"),
            party_kind=kinds.get((axis, dcid)),
        )
        for dcid in members
        if (label := labels.get(dcid)) is not None
    ]
    return OptionsSet(items=items, truncated=False)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComputedOptions:
    """The options `compute_options` built and the warnings it carried.

    `transient` marks a result built from a read that failed during this compute: a coverage
    probe that did not answer, a UN System Data Commons read that failed
    (`UnBlockedPlaces.failures`), or a UN front door refusal of a place outside the bound
    origin catalogue build's refused set. Places in that set are a fact of the build and
    leave a result settled, though its warning names them. `degraded` marks a compute that
    noted a read it stands without, lost or cut by its budget
    (`degraded_reads.DegradedReads.storable`); the caller that records those reads sets it.
    `options_response_cache.is_cacheable` reads both."""

    options: OptionsSet
    warnings: tuple[Warning, ...]
    transient: bool = False
    degraded: bool = False


def compute_options(
    req: OptionsRequest,
    target: ResolvedOptionsTarget,
    *,
    graph: EngineGraphClient,
    deadline: float,
    materialise_graph: GraphReader | None = None,
    materialise_sdmx: MaterialiseSdmxClient | None = None,
    coverage: CoverageSnapshot,
) -> ComputedOptions:
    """Build the requested role's options, their warnings and whether a read failed.

    `observationAbout` reads graph candidates; flow axes group the supplied
    coverage snapshot. `materialise_sdmx` is used only for request-form unit
    conversion. `deadline` is the absolute monotonic deadline the
    observationAbout reads honour. The supplied snapshot is shared across both
    branches. Warnings report coverage-probe failures and UN System Data
    Commons reads that could not serve (`un_system_unavailable`). The target's
    provenance pin scopes every read (`resubmit_request.pinned_provenance_scope`), so a
    base pin answers at a place a DFO table also holds.
    """
    # A provenance pin scopes every read before DFO precedence, as resubmit reads it.
    with recording_un_blocked_places() as un_blocked, pinned_provenance_scope(target.facets):
        if req.for_.role == "observationAbout":
            options, probe_warning = _observation_about_options_set(
                req,
                target,
                graph=graph,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
                include_coverage=bool(req.include_coverage),
                deadline=deadline,
                coverage_snapshot=coverage,
            )
        else:
            options = _directional_options_set(
                target, graph=graph, axis=cast(RoleAxis, req.for_.role), coverage_snapshot=coverage
            )
            probe_warning = None

    # The door holds the population's dcids and no member names, so the warning names
    # its variables `""`.
    warnings = tuple(
        w
        for w in (probe_warning, un_unavailable_warning(un_blocked, labels=NO_LABELS))
        if w is not None
    )
    return ComputedOptions(
        options=options,
        warnings=warnings,
        transient=probe_warning is not None
        or bool(un_blocked.failures())
        or _refused_while_serving(un_blocked.places()),
    )


def _refused_while_serving(places: tuple[str, ...]) -> bool:
    """Whether the UN front door refused a place outside the bound origin catalogue build's
    own refused set (`Catalogue.un_blocked_places`). Such a refusal was found while serving
    and may be a stray one, so the result is transient."""
    return bool(places) and not set(places) <= current_catalogue().un_blocked_places


def build_options_response(
    req: OptionsRequest,
    target: ResolvedOptionsTarget,
    *,
    graph: EngineGraphClient,
    deadline: float,
    materialise_graph: GraphReader | None = None,
    materialise_sdmx: MaterialiseSdmxClient | None = None,
    coverage: CoverageSnapshot,
) -> OptionsResponse:
    """`compute_options` in the response envelope. The diagnostics time this call alone;
    `timing_by_step` is empty."""
    start_ms = now_ms()
    computed = compute_options(
        req,
        target,
        graph=graph,
        deadline=deadline,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=coverage,
    )
    diagnostics = make_diagnostics(build_commit(), list(computed.warnings), {}, now_ms() - start_ms)
    return OptionsResponse(diagnostics=diagnostics, for_=req.for_, options=computed.options)
