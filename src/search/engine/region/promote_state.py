"""Per-call state for the standard-promote phase sequence.

Phase modules import this leaf module to avoid importing the sequence driver."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, cast

from code_index.query import IndexFamily
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from materialise import GraphReader
from materialise.facets import FacetSelection
from materialise.place_scope import DefaultPresence
from materialise.rate_tables import RateTableCache
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.departures import WarningKey, warning_key
from search.engine.extract import DateRequest
from search.engine.region.convertibility import UnitCoverage
from search.engine.region.place_context import PlaceContext
from search.engine.region.result import FacetAxisFilter
from search.engine.retrieval.bound_values import (
    MeasurePartition,
    OpenAxisFact,
    OptionServedFact,
    RoleClassWalks,
    WalkedMembers,
)
from search.engine.retrieval.facts import FiveTuple
from search.engine.retrieval.read_plan import ReadPlan
from search.engine.retrieval.retrieve import (
    Materialised,
    MaterialisedCandidates,
    NoDataDraft,
    UnsupportedDraft,
)
from search.engine.shape import ShapeDraft
from search.engine.spec_build import RoleAxisFactInput
from search.engine.unit.bind import RequestBasis
from search.heading.departure_fact import DepartureFact
from search.models import (
    Coverage,
    Entity,
    EntityOptions,
    FacetBasis,
    Headline,
    ProvenanceRef,
    ResubmitIdentityPayload,
    Shape,
    Slot,
    SlotOptions,
    Spec,
    SpecObservations,
    StatVar,
    Warning,
)
from series_coverage.shared import CoverageSnapshot

if TYPE_CHECKING:
    # TYPE_CHECKING avoids the runtime cycle through options.population and resubmit_promote.
    from search.engine.options.population_memo import OptionsPopulationMemo
    from search.engine.region.option_coverage import RoleServedFilter

    # region.place_options sits downstream of the exact same cycle (see
    # region/resubmit_promote.py's own _relaunch_place_options closure for
    # the full chain), so PlaceOptionsPool is deferred here for the same
    # reason.
    from search.engine.region.place_options import PlaceOptionsPool
    from search.engine.region.related_read import PendingRelated
    from search.engine.retrieval.member_facts import MemberFacts


@dataclasses.dataclass(slots=True, kw_only=True)
class _PromoteState:
    """Request context is fixed at construction; later fields hold phase outputs."""

    # --- (a) request context -------------------------------------------------
    inp: ResubmitIdentityPayload
    graph: EngineGraphClient
    sdmx: EngineSdmxClient
    deadline: float | None
    include_option_coverage: bool
    stream_slot_options: bool
    include_observations: bool
    stream_observations: bool
    materialise_graph: GraphReader | None
    materialise_sdmx: MaterialiseSdmxClient | None
    rate_cache: RateTableCache | None = None
    population_memo: OptionsPopulationMemo | None = None
    place_options_pool: PlaceOptionsPool | None = None
    # Shared coverage snapshot for materialise, unit decisions, and observations; required.
    coverage: CoverageSnapshot

    # --- (b) evolving state ----------------------------------------------------
    # Initial display text; the shape phase assigns the verified slug.
    variable_text: str = ""
    # The members the identity elects (`promote_reread._elected_members`).
    stat_var_dcids: list[str] = dataclasses.field(default_factory=list)
    # Constraint properties forced by a named-family election. Empty for any other election
    # and disjoint from the posted bindings.
    measure_elected_properties: frozenset[str] = frozenset()
    # The posted identity's scalar observationAbout anchor dcid (the first confirmed ref).
    entity_dcid: str | None = None
    # Re-read arcs for every elected SV, keyed by dcid.
    confirmed_arcs: dict[str, dict] = dataclasses.field(default_factory=dict)
    # Shared five-tuple, set after arc confirmation; the cast marks its phase-dependent value.
    ft: FiveTuple = dataclasses.field(default=cast(FiveTuple, None))
    # Exact qualifier values seen on the elected family's graph arcs. This
    # retains dangling dcids that read_five_tuple omits from public identity.
    family_measurement_qualifiers: frozenset[str | None] = frozenset()
    # Display slug from the shared five-tuple; shape identity uses the regenerated structure.
    regenerated_shape_id: str = ""
    # The index family that decides every elected member's membership, or None when the
    # members span families or the graph walk decides them.
    index_family: IndexFamily | None = None
    # Union of every elected SV's own constraint map, keyed by property to the
    # set of values asserted across the whole elected set.
    union_constraints: dict[str, set[str]] = dataclasses.field(default_factory=dict)
    # What/how properties some elected member asserts outside the requested values, keyed to
    # the requested values.
    edits: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    # Client-controlled what/how properties posted unbound, tracked as widenings. Election-
    # forced properties stay out; the identity leaves them open, with arcs supplying their
    # elected binding.
    widened_props: set[str] = dataclasses.field(default_factory=set)
    # Every posted-bound what/how property, keyed to its requested values,
    # regardless of whether it turned out to be an edit.
    posted_bound_values: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    # Final StatVar set served by this resubmit after any narrowing.
    resolved_stat_var_dcids: list[str] = dataclasses.field(default_factory=list)
    # resolved_stat_var_dcids' own confirmed arcs, and from _phase_build_shape on those of
    # every member of the family the mint resolves.
    resolved_arcs: dict[str, dict] = dataclasses.field(default_factory=dict)
    # The mint's member walks for this request, in their resubmit mode, shared by the member
    # election and the shape's role-class election so each distinct query walks once. None
    # without a materialise graph client.
    role_class_walks: RoleClassWalks | None = None
    # Warning list shared with every phase.
    warnings: list[Warning] = dataclasses.field(default_factory=list)
    # First resolved member, anchoring standard_bindings_from_arcs.
    rep_sv: str = ""
    # Resolved-member arcs used by shape reconstruction.
    sv_arc_facts: dict[str, dict] = dataclasses.field(default_factory=dict)
    # Reconstructed from the shared five-tuple and resolved members' arcs.
    shape_draft: ShapeDraft = dataclasses.field(default=cast(ShapeDraft, None))
    # Posted where-set members (destination or observationAbout), read from identity.entities.
    # Counts above max_standard_places raise in _phase_where_places; accepted values stay intact.
    standard_where_dcids: list[str] = dataclasses.field(default_factory=list)
    # Posted origin refs from identity.entities, normalised by its schema.
    origin_dcids: list[str] = dataclasses.field(default_factory=list)
    # Full posted entity map, retaining roles the resolved shape cannot host for validation.
    identity_entities: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    # Identity's what/how slots adapted once from the elected members' confirmed arcs.
    # Every later phase uses this wire-only view for existing slot helpers.
    posted_slots: list[Slot] = dataclasses.field(default_factory=list)
    # The shape's place context from _phase_where_places: its role entities, wire roles, probe
    # places and any default scope. _phase_materialise settles it on a discovered scope or on
    # the default scope of the StatVars the answer serves. The cast marks its
    # phase-dependent value.
    place: PlaceContext = dataclasses.field(default=cast(PlaceContext, None))
    # The presence read a default scope comes from; None when the shape binds a place.
    default_presence: DefaultPresence | None = None
    # where + arc-derived constraint bindings for this resubmit's answer.
    bindings: list[SlotBindingDraft] = dataclasses.field(default_factory=list)
    # Resolved from identity.requested_window; re-resolves "latest" to the
    # current latest year.
    date_request: DateRequest | None = None
    # Whether the index or the mint's walk settled the family the shape holds, so this
    # resubmit elects among it as the first answer does (`_phase_build_shape`).
    family_settled: bool = False
    # What the mint states of the members the walk settled for this resubmit's identity,
    # `None` when no walk settled them (`MembershipResolution.walked`).
    walked: WalkedMembers | None = None
    # The standard resolver's own materialise() outcome for this resubmit.
    mat_result: Materialised | NoDataDraft | MaterialisedCandidates | UnsupportedDraft | None = None
    # Shared timing dict populated by every timed step.
    timing: dict[str, int] = dataclasses.field(default_factory=dict)
    # The posted source-axis constraint's ref dcid(s) (slot or facets.provenance).
    source_dcids: list[str] = dataclasses.field(default_factory=list)
    # _build_observations' facet_filter, built from source_dcids and
    # identity.facets.
    facet_filter: Mapping[str, FacetAxisFilter] | None = None
    # The memoised opposite-role label lookup the materialised warnings were built with
    # (`grounding.memoised_labels`); grounding restates them through it.
    opposite_labels: Callable[[Sequence[str]], Sequence[str]] | None = None
    # Coverage, the headline and the preview read plan under `facet_filter` and the elected
    # units (`region.facets.filter_answer`). The plan also names the sources, dates each
    # StatVar and lists the scope members.
    narrowed_coverage: Coverage = dataclasses.field(default=cast(Coverage, None))
    narrowed_headline: Headline | None = None
    narrowed_plan: ReadPlan | None = None
    # What the elected conversion does to the rows the narrowed plan serves
    # (`convertibility.convertible_answer`): the plan, coverage and headline above already
    # leave out the rows it omits, and the bound-unit gate reports its refusal or warning.
    unit_coverage: UnitCoverage = dataclasses.field(default_factory=UnitCoverage)
    # The plan the bound-unit gate reads (`convertibility.unit_gate`): the filtered plan, less
    # what a `latest` read omits before it takes each StatVar's latest period, until the
    # conversion narrows the answer's own plan. None when the gate is never reached.
    unit_gate_plan: ReadPlan | None = None
    # The unit refs a bound unit request's convertibility decision elected,
    # per stat_var (region/convertibility.py's elected_unit_refs).
    # Non-None only when state.elected_unit is itself non-None. Read by
    # _phase_ground_build's own facet-narrowing step and by its final
    # facet_refusal gates.
    elected_units: Mapping[str, frozenset[str]] | None = None
    # Grounded Slot objects from _ground_answer's unpack, later rebound by
    # bind_when_slot and _rebind_source_slot.
    slots: list[Slot] = dataclasses.field(default_factory=list)
    # The resolved data-source refs from the same _ground_answer unpack.
    source_refs: list[ProvenanceRef] = dataclasses.field(default_factory=list)
    # The same unpack's label batch, which `facets.answer_facets` reads.
    labels: dict[str, str] = dataclasses.field(default_factory=dict)
    # _ground_answer's own warnings, kept separate from `warnings` until the
    # definite return stamps them with spec.spec_id.
    ground_warnings: list[Warning] = dataclasses.field(default_factory=list)
    # The departure facts raised beside warnings this resubmit made (`depart`), each under
    # its warning's key. The resolved region holds them (`resubmit_promote`), and a part
    # keeps one while it keeps the warning.
    departure_facts: dict[WarningKey, DepartureFact] = dataclasses.field(default_factory=dict)
    # Property dcids among `slots`' constraint slots that no resolved
    # StatVar asserts, recomputed from resolved_arcs.
    inapplicable_slots: list[str] = dataclasses.field(default_factory=list)
    # Same arc-based source as inapplicable_slots, reporting the
    # present-but-unresolved state directly.
    unresolved_slots: list[str] = dataclasses.field(default_factory=list)
    # The grounded Shape model built from shape_draft and _ground_answer's
    # grounded five-tuple/slot refs. Same not-yet-run placeholder rationale
    # as ft above.
    shape_model: Shape = dataclasses.field(default=cast(Shape, None))
    # The grounded StatVar objects built from sv_refs and resolved_arcs.
    stat_vars: list[StatVar] = dataclasses.field(default_factory=list)
    # Entity objects assembled from _ground_answer's grounded entity data.
    entities: list[Entity] = dataclasses.field(default_factory=list)
    # The role -> entity-dcid-list map to probe for observation points.
    obs_entities: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    # The warning for an unplaced observation role, derived after grounding. Kept apart from
    # `warnings` so refusals order it after the ground warnings, as a fresh request does.
    obs_entities_warning: Warning | None = None
    # The role -> entity-dcid-list map coverage probes read (facet_options), with
    # discovered-scope members left out.
    probe_role_entities: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    # probe_role_entities flattened to the single entity list facet_refusal/
    # slot_options coverage expect.
    flat_probe_entities: list[str] = dataclasses.field(default_factory=list)
    # Baseline slot options from grounding, enriched after a definite Spec is confirmed.
    slot_options: list[SlotOptions] = dataclasses.field(default_factory=list)
    # The assembled Spec for this resubmit's answer. No empty Spec exists to
    # default to, and every reader runs after build_spec. Same not-yet-run
    # placeholder rationale as ft above.
    spec: Spec = dataclasses.field(default=cast(Spec, None))
    # Served members' measure partition and siblings, retained for RegionResult and merges.
    measure_partition: MeasurePartition | None = None
    measure_siblings: tuple[MeasurePartition, ...] = ()
    # The related-measures reads started as grounding starts (`related_read`), which
    # finalisation joins and `_resolve_standard_promote` waits for on every exit. None when the
    # answer does not qualify.
    related_read: PendingRelated | None = None
    # Open constraint axes and served-member values, published on RegionResult.
    open_axis_facts: tuple[OpenAxisFact, ...] = ()
    # What `slot_options` was counted from, published on RegionResult for merges.
    option_axis_facts: tuple[OptionServedFact, ...] = ()
    # The served test and served members, published on RegionResult for merges.
    served_filter: RoleServedFilter | None = None
    served_members: tuple[MemberFacts, ...] = ()
    # Open role axes feed both Spec and entity-option construction from one derivation.
    role_axis_facts: tuple[RoleAxisFactInput, ...] = ()
    # Facet axes bound_facet_axes(slots, identity.facets) confirms as bound.
    bound_facet_axes: dict[str, FacetBasis] = dataclasses.field(default_factory=dict)
    # The IdentityUnitRequest basis derived from identity.facets, when a unit
    # request is bound.
    unit_request_basis: dict[str, RequestBasis] | None = None
    # decide_part_unit's own per-stat_var selection map for a bound
    # request-form unit (region/convertibility.py), kept so _phase_finalize
    # can pass it straight to _build_observations/_observations_relaunch_closure
    # as `elected=`, steering the preview onto the same facet(s)
    # _phase_ground_build already confirmed convertible. None when no unit
    # request was bound, or role bindings were empty during grounding.
    elected_unit: Mapping[str, tuple[Mapping[str, FacetSelection], ...]] | None = None
    # The eagerly-fetched observations preview, when stream_observations,
    # include_observations, or a bound facet filter with a resolved window
    # requires one.
    fetched_obs: SpecObservations | None = None
    # Raw entity options used by the slot-options relaunch; populated during finalisation.
    entity_options_raw: list[EntityOptions] | None = None

    def depart(self, warning: Warning, fact: DepartureFact) -> Warning:
        """Record `fact` as the departure `warning` discloses, and return `warning` for the
        caller to place among its warnings."""
        self.departure_facts[warning_key(warning)] = fact
        return warning

    def refusal_warnings(self) -> tuple[Warning, ...]:
        """The warnings a refusal carries, as a fresh request's refusal carries them for the
        same evidence: a refusal holds no Spec, so ground warnings stay unstamped, and they
        precede the observation-entities warning."""
        return (
            *self.warnings,
            *self.ground_warnings,
            *((self.obs_entities_warning,) if self.obs_entities_warning is not None else ()),
        )

    def definite_warnings(self, spec_id: str) -> tuple[Warning, ...]:
        """The warnings that precede finalisation of a definite answer, with ground warnings
        stamped with `spec_id`."""
        return (
            *self.warnings,
            *((self.obs_entities_warning,) if self.obs_entities_warning is not None else ()),
            *(w.model_copy(update={"spec_id": spec_id}) for w in self.ground_warnings),
        )
