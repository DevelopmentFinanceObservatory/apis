"""Per-call state shared by the Search resolve stages.

Request context is fixed at construction; pipeline state accumulates as stages
run. Both `region.pipeline` and `region.candidates` import the type here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, cast

from code_index.members import FamilyQuery
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from dfo_platform import Cause
from entity_resolution import Admission, EntityResolver, Resolution
from materialise import GraphReader
from materialise.facets import FacetSelection
from materialise.place_scope import DefaultPresence
from materialise.rate_tables import RateTableCache
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.dataset_mention import DatasetMention
from search.engine.departures import WarningKey, warning_key, window_gap_fact
from search.engine.envelope import make_pipeline_step
from search.engine.events import EventSink
from search.engine.extract import DateRequest
from search.engine.interpret import Recall
from search.engine.qualifier_match import (
    CodedQualifier,
    QualifierMatch,
    QualifierReadings,
    QualifierSpan,
    QualifierTie,
)
from search.engine.region.convertibility import UnitCoverage
from search.engine.region.materialise_step import Departed, family_total_absent_departures
from search.engine.region.observations.entities import open_role_axis
from search.engine.region.place_context import PlaceContext, empty_place_context
from search.engine.region.place_roles import PlaceRoles
from search.engine.region.result import (
    AmbiguousEntity,
    FacetAxisFilter,
    RegionResult,
    RoleNotAdmittedEntity,
)
from search.engine.retrieval.bound_values import (
    MeasurePartition,
    RoleClassWalks,
    entity_role_keys,
    identity_query,
    index_decides_membership,
    query_bound_values,
    shape_member_facts,
    shape_query,
)
from search.engine.retrieval.facts import read_name
from search.engine.retrieval.read_plan import ReadPlan, minimal_constraint_svs
from search.engine.retrieval.retrieve import (
    FamilyTotalAbsent,
    Materialised,
    MaterialisedCandidates,
    NoDataDraft,
    UnsupportedDraft,
)
from search.engine.retrieval.role_resolve import role_members_served
from search.engine.role_text import DoorRole
from search.engine.shape import ShapeDraft, is_role_bearing, role_names_by_axis
from search.engine.unit.bind import BoundUnitRequest, RequestBasis
from search.heading.departure_fact import DepartureFact
from search.models import (
    BaseIdentity,
    Coverage,
    Entity,
    EntityRoleKey,
    FacetBasis,
    GraphRef,
    Headline,
    IdentityUnitRequest,
    NearestRealSpec,
    NoDataGate,
    Operation,
    PipelineStep,
    PipelineStepName,
    ProvenanceRef,
    ResolvedFacet,
    ResolvedQualifier,
    Shape,
    Slot,
    SlotOptions,
    Spec,
    StatVar,
    TraceDefaultReading,
    UnitAlternative,
    Warning,
)
from series_coverage.shared import CoverageSnapshot

if TYPE_CHECKING:
    # Deferred to break the runtime cycle through `resubmit_promote` and this module.
    from search.engine.options.population_memo import OptionsPopulationMemo
    from search.engine.region.alternate_guard import AlternateJudge

    # `candidates` imports this module.
    from search.engine.region.candidates import _Alternate, _Candidate

    # Shares the same runtime cycle, so this import is type-checking only.
    from search.engine.region.place_options import PlaceOptionsPool
    from search.engine.region.related_read import RelatedRead

__all__ = ["EngineOptions", "ResolveState", "raised_facts"]


# A served measure partition and its sibling partitions (`bound_values.observed_measure_family`).
type MeasureFamily = tuple[MeasurePartition | None, tuple[MeasurePartition, ...]]


def raised_facts(departed: Iterable[Departed]) -> dict[WarningKey, DepartureFact]:
    """Each departure fact in `departed` under the key of the warning that discloses it."""
    return {warning_key(warning): fact for warning, fact in departed}


@dataclass(frozen=True, slots=True, kw_only=True)
class EngineOptions:
    """Optional enrichments and clients for resolve and resubmit calls.

    Defaults preserve the baseline answer without enrichment.
    """

    # Probe each returned Spec's slot options and embed coverage (one graph call per slot).
    include_option_coverage: bool = False
    # Probe slot coverage in the background; the caller awaits the task after yielding the response.
    # Independent of include_option_coverage and leaves returned slot options unchanged.
    stream_slot_options: bool = False
    # Embed observations for definite Specs, using full small series or a bounded latest-points preview.
    include_observations: bool = False
    # Fetch observations in the background, independently of include_observations.
    stream_observations: bool = False
    # GraphReader passed to each observations-preview call; `None` is forwarded as-is.
    materialise_graph: GraphReader | None = None
    # SdmxClient passed to observations-preview calls alongside `materialise_graph`.
    materialise_sdmx: MaterialiseSdmxClient | None = None
    # Rate table cache passed to observations-preview calls; `None` is forwarded as-is.
    rate_cache: RateTableCache | None = None
    # Request-scoped memo passed to `build_place_options_events`; `None` skips the lookup.
    population_memo: OptionsPopulationMemo | None = None
    # Request-scoped pool passed with `population_memo`; `None` skips place-option tasks.
    place_options_pool: PlaceOptionsPool | None = None
    # Monotonic deadline for graph and SDMX reads; `None` means unbounded.
    deadline: float | None = None


# Terminal step names and wire order; mark_remaining_not_ran adds missing, unrecorded steps.
_TERMINAL_STEPS: tuple[PipelineStepName, ...] = ("shape", "bind", "materialise", "answer")


def _bound_role_dcids(
    shape: ShapeDraft, resolved_entities: Mapping[str, Sequence[str]]
) -> dict[EntityRoleKey, tuple[str, ...]]:
    """Map each bound wire role to its dcids; subject-only shapes return `{}`.

    Unlike `open_role_axis`, this records every bound axis, including both
    origin and destination, so no-data guards can preserve query direction.
    """
    if not is_role_bearing(shape):
        return {}
    bound: dict[EntityRoleKey, tuple[str, ...]] = {}
    for axis, name in role_names_by_axis(shape).items():
        dcids = resolved_entities.get(name)
        if dcids:
            bound[axis] = tuple(dcids)
    return bound


def _elected_stat_vars(
    mat_result: Materialised | MaterialisedCandidates | NoDataDraft | UnsupportedDraft | None,
    shape_draft: ShapeDraft,
    *,
    allow_shape_draft_fallback: bool = False,
) -> list[GraphRef] | None:
    """Return confirmed StatVars from the materialise result, or `None`.

    Only named-entity and role-incompatible terminals may opt into the
    ShapeDraft's confirmed-member fallback. Labels use each member's `->name`
    arc, falling back to its dcid.
    """
    sv_dcids: list[str] | None = None
    if isinstance(mat_result, (Materialised, NoDataDraft)):
        sv_dcids = mat_result.sv_dcids
    if not sv_dcids:
        # This role refusal can precede a surviving-member list; the elected shape
        # was already graph-confirmed, so it can use the terminal fallback.
        materialise_role_refusal = (
            isinstance(mat_result, NoDataDraft) and mat_result.reason == "role_incompatible"
        )
        if (
            not allow_shape_draft_fallback
            or (mat_result is not None and not materialise_role_refusal)
            or not isinstance(shape_draft, ShapeDraft)
        ):
            return None
        sv_dcids = list(shape_draft.sv_arc_facts or ())
    if not sv_dcids:
        return None
    # materialise runs only after bind sets `shape_draft`.
    arc_facts = shape_draft.sv_arc_facts or {}
    return [
        GraphRef(dcid=dcid, label=read_name(arc_facts.get(dcid, {})) or dcid) for dcid in sv_dcids
    ]


def served_stat_vars(
    shape: ShapeDraft, bindings: Sequence[SlotBindingDraft], roles: PlaceRoles
) -> list[str]:
    """The StatVars a Datasets mint serves for the Spec `shape` grounds under `bindings` with
    its places bound as `roles`, in member order (`_served_members` over `shape_query`)."""
    return _served_members(
        shape, shape_query(shape, bindings, entity_role_keys(shape, roles.resolved_entities))
    )


def identity_served_stat_vars(identity: BaseIdentity, shape: ShapeDraft) -> list[str]:
    """The StatVars a Datasets mint serves for a resubmitted `identity`, in member order.

    `shape` holds its members exactly (`ShapeDraft.sv_arc_facts_complete`): the family the
    mint's walk resolves with the posted members as seeds once that walk settles them, and
    the posted members otherwise (`elected_role_class_shape`). The mint resolves an identity
    the code index decides through the index and ignores the posted members
    (`index_decides_membership`), so for that identity the index's family answers."""
    if index_decides_membership(identity):
        shape = replace(shape, sv_arc_facts_complete=False)
    return _served_members(shape, identity_query(identity))


def _served_members(shape: ShapeDraft, query: FamilyQuery) -> list[str]:
    """The StatVars a Datasets mint serves for `shape` under `query`.

    The population is the one `shape_member_facts` resolves. For a query the code index
    decides, that is the elected dataset's family (`code_index.members.index_family`), and
    the mint serves its antichain-minimal rows. For a shape the index does not decide, it is
    the walked family once the mint's walk settles the members, which the shape then holds
    exactly (`elected_role_class_shape`), and the recalled or posted members otherwise,
    which the answer flags `membership_unverified`. A role-bearing population is reduced to
    the members the answer lists (`role_resolve.role_members_served`), and any other to its
    antichain-minimal members."""
    member_facts, index_sourced = shape_member_facts(shape, query)
    if is_role_bearing(shape) and not index_sourced:
        return role_members_served(member_facts, query_bound_values(query), query.role_keys)
    return minimal_constraint_svs(
        list(member_facts), {sv: member.read_constraints() for sv, member in member_facts.items()}
    )


# Why the presence read an alternate's default scope needs did not complete: no worker of
# the alternates' presence pool was free, the presence source failed, or it timed out.
AlternateUnevaluatedReason = Literal["no_capacity", "upstream_error", "timed_out"]


@dataclass(slots=True, kw_only=True)
class ResolveState:
    """Per-variable state for the stages extracted out of resolve_variable.

    (a) Request context: resolve_variable's own parameters, set once at
    construction and never reassigned by any stage.
    (b) Evolving pipeline state: produced by earlier stages and read (some
    also written) by later ones. Optional/late-bound where a stage may not
    have run yet.
    """

    # --- (a) request context -------------------------------------------------
    variable: str
    entities: list[str]
    date_request: DateRequest | None
    detect_query: str
    role_query: str
    # Door-supplied roles per entity name, in mention order. A self-flow can
    # list both axes; an absent name has no role and uses the default handling.
    entity_roles: dict[str, list[DoorRole]] = field(default_factory=dict)
    # Extracted dropped phrases shared across every variable leg, followed on the
    # raw-text door by this leg's own operation phrases; each binder checks them
    # against its shape, then warnings combine admission across regions.
    dropped_qualifier_phrases: list[str] = field(default_factory=list)
    # This leg's admitted operations. One over `category` opens the slots it names on
    # the elected shape (`phase_rank_bind._open_operation_slots`).
    operations: tuple[Operation, ...] = ()
    # Verbatim query for qualifier matching; set only for a single-variable
    # raw-text request, where query-level phrases have an unambiguous owner.
    query_text: str | None = None
    # Typed qualifier sidecar for this variable, populated by raw-text extraction.
    # Its spans address each phrase, not `variable`; `detect_query` includes its recall text.
    qualifiers: tuple[CodedQualifier, ...] | None = None
    pac: bool
    graph: EngineGraphClient
    resolver: EntityResolver
    admission: Admission
    llm: SupportsLLM
    sdmx: EngineSdmxClient
    deadline: float | None
    pre_resolved: dict[str, Resolution] | None
    variable_index: int
    emit: EventSink | None
    include_option_coverage: bool
    stream_slot_options: bool
    include_observations: bool
    stream_observations: bool
    unit_text: str | None
    # The raw-text door's unit phrase is best effort: data that cannot convert answers
    # unconverted, with the phrase departing (`phase_shared.withdraw_typed_unit`).
    unit_text_best_effort: bool = False
    source_text: str | None
    materialise_graph: GraphReader | None
    materialise_sdmx: MaterialiseSdmxClient | None
    rate_cache: RateTableCache | None = None
    population_memo: OptionsPopulationMemo | None = None
    place_options_pool: PlaceOptionsPool | None = None
    # The mint's member walks this request has run, shared by its passes and alternates.
    # None when no member graph is supplied, and no walk runs.
    role_class_walks: RoleClassWalks | None = None
    # One request snapshot shared by materialisation, unit selection, and observations.
    # Required so consumers cannot take inconsistent coverage-service reads.
    coverage: CoverageSnapshot
    # The recall pool shapes derive from: our candidates, or the UN System Data Commons'
    # on the gap-fill pass that runs when ours end no_data (`region.pipeline`).
    pool: Literal["ours", "un_system"] = "ours"

    # --- (b) evolving pipeline state ------------------------------------------
    pipeline_steps: list[PipelineStep]
    # The timed pipeline steps recorded so far; read by the terminal builders
    # to mark the remaining steps not-ran.
    ran_steps: set[PipelineStepName] = field(default_factory=set)
    timing: dict[str, int]
    # Shared warnings list; mutations here affect every stage holding the list.
    warnings: list[Warning]
    # The departure facts raised beside warnings, each under its warning's key
    # (`departures.warning_key`). A part keeps those its final warnings still hold.
    departure_facts: dict[WarningKey, DepartureFact] = field(default_factory=dict)
    # By `spec_id`, each grounded Spec whose single StatVar is not its family's total.
    # A terminal raises the warning for the Specs its part holds.
    family_total_absent: dict[str, FamilyTotalAbsent] = field(default_factory=dict)
    # Accumulates LLM token usage for this variable's pipeline calls.
    var_usage: dict | None = None
    # Keep a place-less `no_matching_slice` deferral on the single-shape path,
    # where minting can attach its required `flagged_match`.
    force_single_shape_path: bool = False
    # Set when a bound, expressible unit request should narrow this
    # variable's observation facets; None for every other request (the
    # overwhelmingly common case).
    bound_unit_outcome: BoundUnitRequest | None = None
    unit_facet_filter: dict[str, FacetAxisFilter] | None = None
    # Parsed unit identity, used only when `bound_unit_outcome` is set.
    unit_request: IdentityUnitRequest | None = None
    bound_facet_axes: dict[str, FacetBasis] = field(default_factory=dict)
    unit_request_basis: dict[str, RequestBasis] | None = None
    # Per-StatVar unit election from `_phase_ground_answer`; later scope and
    # observation reads reuse it to avoid repeating the graph/SDMX lookup.
    unit_elected_selections: Mapping[str, tuple[Mapping[str, FacetSelection], ...]] | None = None
    # The per-variable recall() result: candidate stat-vars and the
    # Country-level entity names recall() alone resolved. None before recall runs.
    rcl: Recall | None = None
    # Pre-expansion Country dcids from recall; derive_shapes uses this original set.
    recall_only_place_dcids: list[str] | None = None
    # Place-set name to expanded member dcids for observation probes; names
    # also identify sets without their own dcid.
    place_children_by_name: dict[str, list[str]] = field(default_factory=dict)
    # Settled Group names to member dcids for where-slot binding; members use
    # `SEARCH_MAX_CHILD_PLACES`, not the standard-place cap.
    group_member_dcids_by_name: dict[str, list[str]] = field(default_factory=dict)
    # Settled Group names to the group's own graph node, when the resolver returned one.
    group_dcid_by_name: dict[str, str] = field(default_factory=dict)
    # Graph labels of those nodes, read in one batch at intake. A node the graph gives no
    # label is absent, and its group keeps the query's name for it.
    group_label_by_dcid: dict[str, str] = field(default_factory=dict)
    # Group-member dcids to resolver names; the query names the group, not each member.
    group_member_names_by_dcid: dict[str, str] = field(default_factory=dict)
    # Pre-filter recall count of the pool in use, excluding Topics, for diagnostics.
    n_recalled: int = 0
    # Start tick for the shape timing span, including derivation and ranking.
    shape_t0: int = 0
    # All confirmed shapes, before eligibility and ranking filters.
    shapes_found: list[ShapeDraft] = field(default_factory=list)
    # Whether any confirmed group has directional roles, independent of the elected shape.
    role_bearing_recalled: bool = False
    # Top-K eligible shapes, after score-floor and denominator-availability pruning.
    eligible_shapes: list[ShapeDraft] = field(default_factory=list)
    # Empty measure-head shapes the availability prune dropped, each with the sibling holding
    # data that stayed; the election warns `elected_shape_no_data` when bind elects a sibling.
    measure_head_substitutions: list[tuple[ShapeDraft, ShapeDraft]] = field(default_factory=list)
    # Confirmed shapes the score floor dropped from eligibility, for the gap-fill log line.
    n_below_score_floor: int = 0
    # Shapes past the floor whose population type's name names another measure class than
    # the query, for the gap-fill log line.
    n_measure_class_refused: int = 0
    # Ranked by recall or topic-root score, then breadth and shape id; measure-family
    # siblings of the indexed default are collapsed before the bind cap.
    ranked_shapes: list[ShapeDraft] = field(default_factory=list)
    # Bind prompt's top-K shapes; `chosen_shape_index` indexes this ranked list.
    capped_shapes: list[ShapeDraft] = field(default_factory=list)
    # Recognised qualifiers from the variable, query text, and dropped phrases.
    # Computed once before measure collapse and reused for narrowing and admission.
    query_qualifier_matches: tuple[QualifierMatch, ...] = ()
    # All recognised spans in `variable`, including ambiguities, used to build
    # `ResolvedQualifier.named`. Raw `query_text` is excluded: span recognition
    # accepts ambiguous terms that admission rejects, which would trigger a false
    # `qualifier_not_bound` refusal. Query-text matches still reach admission.
    variable_qualifier_spans: tuple[QualifierSpan, ...] = ()
    # Normalised phrases matched in `variable` or `query_text`, plus tie phrases.
    # Excludes dropped-only phrases, whose warning ownership belongs to raw-text entry.
    variable_matched_qualifier_phrases: frozenset[str] = frozenset()
    # The dataset mention the request's reading bound to this variable, the one thing a
    # source binds from. Vetting writes it into the bindings as the source binding.
    dataset_mention: DatasetMention | None = None
    # Ambiguous qualifier matches across variable, single-variable query text, and
    # dropped phrases. Bind narrows them to one value or alternatives.
    query_qualifier_ties: tuple[QualifierTie, ...] = ()
    # Spans naming one code row and one measure value, from the same sources as the ties.
    # Each reading is admitted only against a shape that carries it alone.
    query_qualifier_readings: tuple[QualifierReadings, ...] = ()
    # Whether a dropped qualifier phrase uses a price-basis word no match read
    # (`alternate_guard.leaves_price_basis_unread`). The default-reading collapse and the
    # guard's default operation both stand down on such a leg.
    unread_price_basis: bool = False
    # What the default-reading collapse did before bind, which every Spec's trace states.
    default_reading: TraceDefaultReading | None = None
    # Ties with multiple rows after declared-property narrowing; bind sees them as
    # alternatives, and qualifier guards treat them as attempted matches. A single
    # surviving row becomes an admitted qualifier instead.
    admitted_ties: tuple[QualifierTie, ...] = ()
    # Normalised phrases from ties the bind model elected; used to suppress a
    # dropped-qualifier warning once this region resolves the phrase.
    elected_tie_phrases: frozenset[str] = frozenset()
    # One recognised value per measure axis, including basis axes. Free-axis
    # matches steer family election; vetting warns when the elected shape lacks one.
    measure_query_matches: tuple[QualifierMatch, ...] = ()
    # Phrases naming conflicting values on one measure axis; they do not steer
    # election and are warned only when this region owns the phrase.
    measure_qualifier_conflict_phrases: frozenset[str] = frozenset()
    # Matched measure phrases carried by the elected shape; these also count as
    # admitted even when the measure axis is not a slot key.
    satisfied_measure_phrases: frozenset[str] = frozenset()
    # Validated `requested_all` phrases from dropped qualifiers. Invalid claims
    # are demoted to `mentioned`; valid phrases suppress the entry-level warning.
    applied_dropped_qualifier_phrases: frozenset[str] = frozenset()
    # Qualifiers admitted against the capped shapes, then narrowed to the elected shape.
    admitted_qualifiers: tuple[QualifierMatch, ...] = ()
    # Bind assigns this before any reader; `cast(None)` avoids an invalid sentinel or Optional checks.
    bind_result: _UniversalBindOutput = field(default=cast(_UniversalBindOutput, None))
    # Shape selected by the bind call.
    shape_draft: ShapeDraft = field(default=cast(ShapeDraft, None))
    # Shape flagged by the near-neighbour gate; the candidate fan-out uses it
    # to distinguish an empty tie-proving alternate from another refusal.
    near_neighbor_fired_index: int | None = None
    # Warning instance retracted by identity if the shape becomes an equivalent-reading annotation.
    near_neighbor_warning: Warning | None = None
    # The chosen shape's own constraint bindings from the universal bind call.
    bindings: list[SlotBindingDraft] = field(default_factory=list)
    # Open constraint axes the elected measure's members do not declare; added to
    # `inapplicable_slots` because they are absent from the shape's slot keys.
    measure_dropped_axes: tuple[str, ...] = ()
    # Bind-time measure election used to label the first sentence event. The
    # materialised rows elect the served partition independently.
    measure_elected_partition: MeasurePartition | None = None
    # The served partition and its siblings (`bound_values.observed_measure_family`), read
    # once grounding starts. `None` before that.
    measure_family: MeasureFamily | None = None
    # The related-measures reads started once materialisation names the elected shape's
    # members (`region.related_read`), which a definite answer takes. `None` when none
    # qualifies.
    related_read: RelatedRead | None = None
    # Every related-measures read the pass started, including ones a later read replaced in
    # `related_read`. The pass waits for them all before it returns (`pipeline._run_phases`).
    related_reads: list[RelatedRead] = field(default_factory=list)
    # Measure properties inserted by family election; `build_spec` marks their provenance.
    measure_elected_properties: frozenset[str] = frozenset()
    # Properties defaulted by `apply_declared_defaults`; `build_spec` marks their provenance.
    defaulted_properties: frozenset[str] = frozenset()
    # Normalised dropped phrases a named slot default answers ("flows" when the flow
    # defaults to ODA); the part's `slot_defaulted` warning replaces their entry-level one.
    slot_default_phrases: frozenset[str] = frozenset()
    # Each named narrowing term and its outcome; build_spec records these, and
    # coverage and final assembly use them for their guards.
    query_qualifiers: list[ResolvedQualifier] = field(default_factory=list)
    # Normalised texts of the terms in `query_qualifiers` whose values matched no member
    # together with the other bound values, so the answer reads without them
    # (`unrealisable.unbind_unrealisable`). Each has its `measure_not_applied` departure,
    # so no qualifier refusal or second departure names it.
    unapplied_qualifier_phrases: frozenset[str] = frozenset()
    # Named places, capped at SEARCH_MAX_STANDARD_PLACES in query order, then distinct and
    # sorted by dcid (`place_roles.place_order`).
    standard_place_dcids: list[str] = field(default_factory=list)
    # The first of `standard_place_dcids` by dcid, for paths that require one place.
    destination_dcid: str | None = None
    # Accepted query labels for resolved dcids that survive the place cap; grounding uses
    # them for display, while identity remains keyed by dcid.
    query_entity_text_by_dcid: dict[str, str] = field(default_factory=dict)
    # Caller-pinned dcids are type-checked only; their text cannot replace graph labels.
    pinned_entity_dcids: frozenset[str] = frozenset()
    # The place context of the shape in `shape_draft`: its roles, the places it is probed
    # and read over, and its place warnings. `_install_shape` sets it with the shape; once
    # the elected result stands it is the context the answer is read over
    # (`PlaceContext.settled`).
    place: PlaceContext = field(default_factory=empty_place_context)
    # Where the StatVars of the last default-scope read have data; the default scope of the
    # shape in play, when it has one, is taken from it. `None` until a shape needs one.
    default_presence: DefaultPresence | None = None
    # The elected shape's binding of the query's places as admitted, before the role-class
    # election narrows it. An alternate binds a place named with no direction on the axis
    # this binding gives it (`place_roles._undirected_place_axis`).
    elected_place_roles: PlaceRoles | None = None
    # Each alternate's binding of the query's places, by capped-shape index, and why each
    # alternate that takes none is left out. `_phase_place_bindings` binds them once.
    alternate_place_roles: dict[int, PlaceRoles] = field(default_factory=dict)
    unbindable_alternates: dict[int, str] = field(default_factory=dict)
    # The capped-shape indices of the siblings zero-observation re-election tries when the
    # elected shape has no observations at its places (`_reelection_siblings`).
    reelection_siblings: list[int] = field(default_factory=list)
    # The presence read a role-bearing elected shape's alternates need, running beside its
    # materialisation (`_start_alternates_presence_read`). Candidate admission awaits it;
    # `None` when none was started or once it is taken.
    alternates_presence: Future[DefaultPresence] | None = None
    # The listing of Earth's countries started beside the elected shape's walk on a query
    # naming no place (`phase_place_materialise._start_countries_listing`). The first
    # default-scope presence read takes it; `None` when none was started or once it is taken.
    # One untaken when the pass ends is cancelled (`pipeline._cancel_queued_reads`).
    countries_listing: asyncio.Future[DefaultPresence] | None = None
    # The same listing when it runs on the alternates' pool for a role-bearing elected
    # shape's alternates (`phase_place_materialise._start_countries_listing`); their
    # presence read, on that pool, waits for it. `None` otherwise. It never queues, since
    # it is not started while that pool is busy (`_submit_alternates_read`).
    alternates_listing: Future[DefaultPresence] | None = None
    # The walks of the alternates and re-election siblings, started beside the elected
    # shape's walk and materialisation and keyed by alternate and representative
    # (`phase_place_materialise._start_alternate_walks`). A walk from a superseded plan stays
    # here until the pass ends. A walk still queued when its pass ends is cancelled, and a
    # running one finishes with its result dropped (`pipeline._cancel_queued_reads`).
    alternate_walks: dict[tuple[object, ...], Future[_Alternate | None]] = field(
        default_factory=dict
    )
    # The alternate guard's judges of the elected shape this pass, one per set of pins
    # (`phase_place_materialise._guard_judge`), so admission, re-election and the walks read
    # each alternate's members once.
    alternate_judges: list[AlternateJudge] = field(default_factory=list)
    # Why the alternates' presence read did not complete, once one has not in this pass
    # (`_read_alternates_presence`). The pass tries no other optional presence read after
    # it, whichever alternates it would have served. The UN System pass starts from a fresh
    # state and reads its own upstream, so a failure here does not stop its reads.
    alternates_presence_failure: AlternateUnevaluatedReason | None = None
    # The alternates that passed the admission guards and were left unevaluated by that
    # failure, by capped-shape index, each with its reason. The result of the pass states
    # them (`pipeline._stating_unevaluated`).
    unevaluated_alternates: dict[int, AlternateUnevaluatedReason] = field(default_factory=dict)
    # The siblings zero-observation re-election materialised with data and offered beside
    # the elected shape, by capped-shape index. Candidate admission takes each from here,
    # so a sibling is materialised once.
    sibling_candidates: dict[int, _Candidate] = field(default_factory=dict)
    mat_result: Materialised | NoDataDraft | MaterialisedCandidates | UnsupportedDraft | None = None
    # Whether the elected shape has data; alternate-shape handling distinguishes
    # candidate competition from reconnaissance with this value.
    elected_has_data: bool = False
    # Elected coverage, narrowed to bound unit facets when present. No empty union arm exists;
    # assigned after answer, like `bind_result`.
    elected_coverage: Coverage = field(default=cast(Coverage, None))
    # The headline and preview read plan under the same facet filter as `elected_coverage`
    # (`region.facets.filter_answer`); assigned after answer.
    elected_headline: Headline | None = None
    elected_plan: ReadPlan | None = None
    # What the elected conversion does to the rows the filtered plan serves
    # (`convertibility.convertible_answer`), computed once before grounding: the plan,
    # coverage and headline above already leave out the rows it omits, and the bound-unit
    # gate reports its refusal or omission warning.
    unit_coverage: UnitCoverage = field(default_factory=UnitCoverage)
    # The plan the bound-unit gate reads (`convertibility.unit_gate`): the filtered plan, less
    # what a `latest` read omits before it takes each StatVar's latest period, until the
    # conversion narrows the answer's own plan. None when the gate is never reached.
    unit_gate_plan: ReadPlan | None = None
    # The memoised opposite-role label lookup the materialised warnings were built with
    # (`grounding.memoised_labels`); grounding restates them through it.
    opposite_labels: Callable[[Sequence[str]], Sequence[str]] | None = None
    slots: list[Slot] = field(default_factory=list)
    source_refs: list[ProvenanceRef] = field(default_factory=list)
    # `resolved.facets`: the facets the answer's read plan serves (`facets.answer_facets`).
    answer_facets: dict[str, ResolvedFacet] = field(default_factory=dict)
    slot_options: list[SlotOptions] = field(default_factory=list)
    # Constraint slots no confirmed StatVar asserts, determined from member arcs.
    inapplicable_slots: list[str] = field(default_factory=list)
    unresolved_slots: list[str] = field(default_factory=list)
    # Built after answer; the type has no empty arm, like `elected_coverage`.
    shape_model: Shape = field(default=cast(Shape, None))
    stat_vars: list[StatVar] = field(default_factory=list)
    entity_objects: list[Entity] = field(default_factory=list)
    # Grounded entities used for spec identity hashing.
    identity_entities: list[Entity] = field(default_factory=list)

    @property
    def tie_unresolved(self) -> bool:
        """Whether a near-neighbour tie fired and an alternate is unevaluated. Unread data is
        no evidence against the other reading, so the tie's warning and its effect on
        confidence stay, whichever answer shape is served."""
        return self.near_neighbor_fired_index is not None and bool(self.unevaluated_alternates)

    @property
    def entity_label_text_by_dcid(self) -> dict[str, str]:
        """Return display aliases for resolved, non-pinned entities."""
        return {
            dcid: text
            for dcid, text in self.query_entity_text_by_dcid.items()
            if dcid not in self.pinned_entity_dcids
        }

    def depart(self, warning: Warning, fact: DepartureFact) -> None:
        """Append `warning` and record `fact`, the departure it discloses, under its key."""
        self.warnings.append(warning)
        self.departure_facts[warning_key(warning)] = fact

    def mark_remaining_not_ran(self) -> None:
        """Append missing terminal steps in wire order, once per step name."""
        already_present = {s.step for s in self.pipeline_steps}
        for step in _TERMINAL_STEPS:
            if step not in self.ran_steps and step not in already_present:
                self.pipeline_steps.append(make_pipeline_step(step, ran=False))

    def label_dcids(self, dcids: Sequence[str]) -> list[str]:
        """Label DCIDs from query text, one batched graph lookup, or the DCID fallback."""
        label_text_by_dcid = self.entity_label_text_by_dcid
        unnamed = [dcid for dcid in dcids if not label_text_by_dcid.get(dcid)]
        graph_labels = self.graph.node_labels_batch(unnamed) if unnamed else {}
        return [label_text_by_dcid.get(dcid) or graph_labels.get(dcid, dcid) for dcid in dcids]

    def family_total_absent_departures(self, specs: Iterable[Spec]) -> list[Departed]:
        """`family_total_absent` and its `closest_match` departure for each of `specs`, the
        Specs a part holds (`materialise_step.family_total_absent_departure`)."""
        return family_total_absent_departures(
            specs, self.family_total_absent, variable_text=self.variable, label_of=self.label_dcids
        )

    def no_data(
        self,
        reason: str,
        *,
        flagged_match: tuple[Spec, ...] | None = None,
        gate: NoDataGate | None = None,
        alternate_match: Spec | None = None,
        nearest_real: tuple[NearestRealSpec, ...] | None = None,
        llm_usage: dict | None = None,
        unresolved_entity_names: tuple[str, ...] = (),
        ambiguous_entities: tuple[AmbiguousEntity, ...] = (),
        resolved_entity_dcids: Mapping[str, str] | None = None,
        role_not_admitted_entities: tuple[RoleNotAdmittedEntity, ...] = (),
        cause: Cause | None = None,
        unbound_qualifiers: tuple[ResolvedQualifier, ...] = (),
        unit_alternatives: tuple[UnitAlternative, ...] = (),
        unit_alternatives_truncated: bool = False,
        candidate: _Candidate | None = None,
    ) -> RegionResult:
        """Build a no-data result, preserving `cause`. A `no_observations` answer over an
        elected result holding data outside the years read carries its `year_unavailable`
        fact (`window_gap_fact`), whatever else the answer names.

        Shape-derived fields (the elected StatVars, the open role axis and the bound role
        dcids) describe `candidate` when the candidates door answers with its refusal, else
        the elected shape.

        The part holds `flagged_match` and `alternate_match`, so those Specs raise
        `family_total_absent`. `elected_stat_vars` names members without a
        Spec and raises none.
        """
        self.mark_remaining_not_ran()
        if candidate is None:
            _shape, _mat_result, _place = self.shape_draft, self.mat_result, self.place
        else:
            _shape, _mat_result, _place = candidate.shape, candidate.mat, candidate.place
        family = self.family_total_absent_departures(
            [*(flagged_match or ()), *([alternate_match] if alternate_match else [])]
        )
        # Report an open role axis only when materialisation's own no-observation
        # result confirms the discovery gap; later refusals may follow successful discovery.
        return RegionResult(
            variable_text=self.variable,
            status="no_data",
            specs=(),
            no_data_reason=reason,
            warnings=(*self.warnings, *(warning for warning, _fact in family)),
            departure_facts={**self.departure_facts, **raised_facts(family)},
            timing_by_step=dict(self.timing),
            flagged_match=flagged_match,
            gate=gate,
            alternate_match=alternate_match,
            nearest_real=nearest_real,
            elected_stat_vars=_elected_stat_vars(
                _mat_result,
                _shape,
                allow_shape_draft_fallback=reason
                in {"entity_not_resolved", "entity_ambiguous", "role_incompatible"},
            ),
            llm_usage=llm_usage,
            unresolved_entity_names=unresolved_entity_names,
            ambiguous_entities=ambiguous_entities,
            resolved_entity_dcids=dict(resolved_entity_dcids or {}),
            role_not_admitted_entities=role_not_admitted_entities,
            cause=cause,
            window_gap_fact=(
                window_gap_fact(_mat_result.window_gap)
                if reason == "no_observations" and isinstance(_mat_result, Materialised)
                else None
            ),
            unbound_qualifiers=unbound_qualifiers,
            unit_alternatives=unit_alternatives,
            unit_alternatives_truncated=unit_alternatives_truncated,
            no_data_role_axis=(
                open_role_axis(_shape, _place.roles.resolved_entities)
                if isinstance(_mat_result, NoDataDraft) and _mat_result.reason == "no_observations"
                else None
            ),
            no_data_bound_role_dcids=(
                _bound_role_dcids(_shape, _place.roles.resolved_entities)
                if isinstance(_shape, ShapeDraft)
                else {}
            ),
        )
