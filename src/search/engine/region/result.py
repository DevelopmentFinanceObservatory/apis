"""Per-variable resolution outcomes and leaf result types for region resolvers."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from dfo_platform import Cause
from materialise.roles import axis_for_role, axis_noun
from search.engine.clients.graph import Facet
from search.engine.departures import (
    NO_ENTITY_WORDS,
    FactsByWarning,
    breakdown_not_applied,
    warning_key,
)
from search.engine.events import PlaceOptionsEvent
from search.engine.region.codes import (
    ALTERNATES_UNEVALUATED,
    QUERY_QUALIFIER_DROPPED,
    _ObsProbeResult,
    _OptionProbeResult,
)
from search.engine.region.facet_probe import FacetProbe
from search.engine.retrieval.bound_values import (
    MeasurePartition,
    OpenAxisFact,
    OptionServedFact,
)
from search.engine.retrieval.member_facts import MemberFacts
from search.engine.retrieval.retrieve import MemberPopulation
from search.engine.role_text import every_member_axis, every_member_breakdown
from search.engine.shape import ShapeDraft
from search.engine.source_code import SourceCodeRefusal
from search.engine.source_scope import AmbiguousSource, SourceRefusal, SourceScope
from search.engine.spec_build import RoleAxisFactInput
from search.heading.departure_fact import DepartureFact
from search.models import (
    Assumption,
    EntityRoleKey,
    GraphRef,
    NearestRealSpec,
    NoDataDenominator,
    NoDataGate,
    NoDataReason,
    Operation,
    Ordering,
    ParsedPart,
    PartScope,
    ResolvedQualifier,
    RoleAxis,
    SlotOptions,
    Spec,
    StatusLiteral,
    UnitAlternative,
    Unsupported,
    Warning,
)
from search.render.assumptions import NameCorrection
from search.render.no_data import UnlistedPopulation

if TYPE_CHECKING:
    from search.engine.bind import SlotBindingDraft
    from search.engine.region.observations.dispatch import PreviewRead
    from search.engine.region.option_coverage import RoleServedFilter
    from search.engine.region.promote_relaunch import PromoteRelaunch
    from search.engine.retrieval.coverage import CoverageForm
    from search.engine.retrieval.read_plan import ReadPlan

FacetPredicate = Callable[[Facet], bool]
"""Test whether a facet passes one axis's filter, including computed filters.

Defined here so `RegionResult` can type it without importing filtering modules.
"""


@dataclass(frozen=True)
class FacetAxisFilter:
    """A facet-axis predicate: whether a facet's value on the axis satisfies the request's
    binding (`region.facets.facet_admitted`)."""

    predicate: FacetPredicate


@dataclass(frozen=True)
class AmbiguousEntityCandidate:
    """One graph node that could match an ambiguous query wording.

    `group_node` is set when the candidate is a group's own node and the part asks for the
    wording's members: a pin on a group's node cannot ask for its members, so the
    candidate's option posts `group_name`, the group's registered name that resolves to it
    alone. A group node with no such name (`group_name` `None`) has no option.
    `group_name` is `None` on every other candidate."""

    dcid: str
    name: str
    kind: str
    group_node: bool = False
    group_name: str | None = None


@dataclass(frozen=True)
class AmbiguousEntity:
    """Query wording with multiple graph candidates.

    `role` is the flow axis this group asks the wording on, or `None` when the door gave
    it none. A wording on both flow ends is one group per end.
    """

    text: str
    candidates: tuple[AmbiguousEntityCandidate, ...]
    role: RoleAxis | None = None


@dataclass(frozen=True)
class RoleNotAdmittedEntity:
    """Resolved entity that no surviving shape can assign to a declared role. A place set
    is named by its node, and one with no node by its first member."""

    text: str
    dcid: str
    # `None` for a set with no node whose members share no stated type.
    graph_type: str | None
    # A whole population asked member by member that no shape lists exactly
    # (`phase_intake._prune_shapes_by_listing`): its side and the classes it leaves out.
    unlisted: UnlistedPopulation | None = None


@dataclass(frozen=True)
class RegionResult:
    """Outcome for one variable, including its Specs and response diagnostics.

    Specs carry pipeline traces. `timing_by_step` records per-step latency.
    Same-shape merges combine regions and union their StatVars.
    """

    variable_text: str | None
    status: StatusLiteral
    specs: tuple[Spec, ...]  # definite: 1, candidates: 2+, no_data/unsupported: 0
    no_data_reason: NoDataReason | None
    warnings: tuple[Warning, ...]
    timing_by_step: dict[str, int]
    earliest_index: int = 0
    # Details for recognized qualifiers with unsupported application rules.
    # Unsupported results return HTTP 200 without a Spec, no-data reason, or warning.
    unsupported: Unsupported | None = None
    # Specs found by relaxation, each with its own relaxation record.
    nearest_real: tuple[NearestRealSpec, ...] | None = None
    # (STANDARD-only) elected Spec with no observation at the resolved place.
    # Contains one Spec and differs from `nearest_real`.
    flagged_match: tuple[Spec, ...] | None = None
    # The dataset a no-data outcome's run settled on (`ResolveState.settled_dataset`),
    # which the restatement of a part that names no source names (`fanout.answered_provenance`).
    # `None` when the run settled on none, and on every other outcome.
    settled_dataset: str | None = None
    # Failed quality-gate code, independent of which near-match field is populated.
    gate: NoDataGate | None = None
    # Unelected data-bearing variable, present only for `only_alternate_has_data`.
    alternate_match: Spec | None = None
    # Confirmed StatVars for the elected shape on no-data outcomes.
    # Matches `flagged_match` on no-member gates. Differs for `only_alternate_has_data`.
    elected_stat_vars: list[GraphRef] | None = None
    # Input entity names the answer could not place, in input order: on a refusal of the
    # entity check (`phase_intake._phase_check_entities`), every name that resolved to no
    # single node, ambiguous ones included, for `role_not_resolved` those whose flow
    # direction the query does not tell (`place_roles.untold_direction_names`), and for
    # `role_incompatible` at that check each whole population directed to the other side
    # (`ResolveState.whole_misdirected`). Empty otherwise.
    unresolved_entity_names: tuple[str, ...] = ()
    # The same `role_not_resolved` names grouped by the place each names, one tuple per
    # untold place set in query order (`place_roles.untold_direction_groups`), so the
    # refusal's sentence counts places. Empty for every other outcome.
    untold_groups: tuple[tuple[str, ...], ...] = ()
    # The graph's label for each place of `untold_groups`, parallel to it
    # (`place_roles.untold_direction_labels`), so the refusal tells apart two places written
    # alike. `None` for a text that resolved to no place. Empty for every other outcome.
    untold_labels: tuple[str | None, ...] = ()
    # Query qualifiers without a final binding, including terms absent from the
    # index vocabulary. Populated only for `qualifier_not_bound`. Full records
    # live on `flagged_match.resolved.trace.query_qualifiers`.
    unbound_qualifiers: tuple[ResolvedQualifier, ...] = ()
    # The sources a `qualifier_not_bound` outcome could not bind, each as an unbound record
    # under its label: the sources the part's scope names or leaves out, or the datasets
    # that alone publish the elected shape. Kept apart from the category terms in
    # `unbound_qualifiers` so the sentence names them as sources.
    unbound_sources: tuple[ResolvedQualifier, ...] = ()
    # The source scope a no-data outcome was produced under (`ResolveState.scope`), set by
    # `pipeline.resolve_variable` on every no-data result of a part with source mentions, so
    # the outcome names the sources that limited it. `None` on every other result.
    source_scope: SourceScope | None = None
    # Ambiguous query wordings and their candidates, on a refusal of the entity check,
    # whatever reason it elects. The wire carries them only for `entity_ambiguous`.
    ambiguous_entities: tuple[AmbiguousEntity, ...] = ()
    # The source wordings a `source_not_resolved` or `source_ambiguous` outcome rests on, in
    # mention order (`source_scope.SourceRefusal.terms`), and the series code a
    # `source_code_*` outcome rests on. Empty for every other outcome.
    source_terms: tuple[str, ...] = ()
    # Each ambiguous source wording with the sources it could name, populated only for
    # `source_ambiguous`, and the series code with the sources holding it for
    # `source_code_ambiguous`.
    ambiguous_sources: tuple[AmbiguousSource, ...] = ()
    # The refusal a `source_code_*` outcome rests on, whose cause, sources and shared codes
    # its sentence names. `None` for every other outcome.
    source_code_refusal: SourceCodeRefusal | None = None
    # Entity names that resolved to exactly one graph node, with that node's dcid,
    # populated only for `entity_ambiguous`. Each of the part's options pins them.
    resolved_entity_dcids: Mapping[str, str] = field(default_factory=dict)
    # Resolved entities without an admitted role in resolution order, populated
    # only for `entity_role_not_admitted`.
    role_not_admitted_entities: tuple[RoleNotAdmittedEntity, ...] = ()
    # Open wire role (`origin` or `destination`) on `no_observations` outcomes
    # from role-bearing shapes, even when scope discovery found no members.
    # The renderer names this axis. Subject-only shapes and other outcomes use `None`.
    no_data_role_axis: EntityRoleKey | None = None
    # Bound role -> entity DCIDs, used to ensure an alternate preserves the
    # query's directional binding when `only_alternate_has_data` fires.
    # Computed from resolved entities after shape construction. `{}` for
    # subject-only shapes or earlier no-data outcomes. This read makes no graph call.
    no_data_bound_role_dcids: Mapping[EntityRoleKey, tuple[str, ...]] = field(default_factory=dict)
    # The answer a refusal raised on behalf of a candidate is about, as its shape, its
    # bindings and the records of the words it leaves unread (`_Candidate.uncovered`): the
    # candidates door answers with that candidate's refusal. `None` for a refusal about the
    # elected reading. A refusal that judged no answer is judged on it
    # (`phase_shared.settle_unjudged_refusal`).
    refused_answer: (
        tuple[ShapeDraft, tuple[SlotBindingDraft, ...], tuple[ResolvedQualifier, ...] | None] | None
    ) = None
    # Stripped, case-folded qualifier spans forced into bindings by this region, answered
    # by a slot default it warns about, or asking for every member of a flow role its
    # answer leaves unbound (`every_member_phrases`). Same-shape merges union the sets.
    # The entry door uses them to retain or drop `QUERY_QUALIFIER_DROPPED` warnings
    # across the request.
    admitted_qualifier_phrases: frozenset[str] = frozenset()
    # The phrases the request asked to leave out that every answer of this region leaves out
    # (`requirements.Judgement.met_exclusions`), as the request wrote them. The entry door
    # drops a phrase's warning on the parts whose regions hold it.
    met_exclusions: frozenset[str] = frozenset()
    # The raw-text door's request-wide `QUERY_QUALIFIER_DROPPED` warnings whose phrase this
    # region leaves unapplied (`raw_text._ExtractionPrep.onto`). The envelope states such a
    # warning only while no region admits its phrase. The part's departures read these
    # beside its own and the envelope's warnings, so a phrase a sibling part applied still
    # departs on this one. No part states them.
    unapplied_qualifier_warnings: tuple[Warning, ...] = ()
    # The flow axes those every-member phrases ask for ("all donors" asks the origin),
    # which the part's assumptions read as asked (`render.assumptions.assumptions_for`).
    # Outside the identity and its hash. Same-shape merges union the sets.
    every_member_axes: frozenset[RoleAxis] = frozenset()
    # The entity names the raw-text door read in place of the words the request typed for
    # them (`raw_text._name_corrections`), request-wide; the part states those its Spec
    # binds (`render.assumptions.assumptions_for`). Outside the identity and its hash. A
    # same-shape merge keeps the first member's, which every member shares. Empty on every
    # other door.
    name_corrections: tuple[NameCorrection, ...] = ()
    # Why a no_data outcome fired (models.NoData's own `cause`), populated
    # only for no_data_reason=="rate_coverage_gap".
    cause: Cause | None = None
    # Capped currencies a `unit_not_resolved` refusal offers. Empty for every other
    # outcome and for a refusal with no candidates.
    unit_alternatives: tuple[UnitAlternative, ...] = ()
    # The readings of a `role_not_resolved` refusal's unstated directions, each mapping
    # every request text of every set whose direction the question leaves untold to one
    # flow axis (`place_roles.direction_readings`). Empty for every other outcome, past
    # `place_roles.MAX_DIRECTION_READINGS` readings, and when no reading holds data.
    role_assignments: tuple[Mapping[str, RoleAxis], ...] = ()
    # The denominator the question named and the one the flagged match divides by, set on a
    # `denominator_not_available` refusal after election. `None` on every other outcome.
    denominator: NoDataDenominator | None = None
    # The `year_unavailable` fact of an answer holding data at the place but none in the
    # years asked (`departures.window_gap_fact`). It holds the years asked and the span the
    # series runs. No warning discloses it, so it rides outside `departure_facts`.
    window_gap_fact: DepartureFact | None = None
    # The departures of a definite answer that no warning discloses: the direction it reads
    # a place named without one on, the one direction that holds data
    # (`departures.direction_inferred`).
    inferred_facts: tuple[DepartureFact, ...] = ()
    # The `whole_total_unpublished` facts of each Spec, by `spec_id`: the whole populations
    # a text names whose side the Spec's dataset reads as unnamed
    # (`whole_totals.whole_total_unpublished_facts`). No warning discloses them.
    unpublished_whole_facts: Mapping[str, tuple[DepartureFact, ...]] = field(default_factory=dict)
    # This region's LLM token usage. A same-shape merge keeps the first
    # member's value. Request-wide totals are aggregated separately.
    llm_usage: dict | None = None
    # True when the candidate budget drops an eligible distinct shape before
    # materialisation. Excludes retrieval top-K and score-floor filtering.
    # Candidate assembly combines this with its own cap for `truncated`.
    candidate_budget_truncated: bool = False
    # The ordering a dataset resubmit posted back with a candidate set, which its part
    # states. A candidate set the status rule builds is `intent_rank`: the elected reading
    # first, then its rivals in menu order. Assembly keeps the specs' order either way.
    candidate_ordering_override: Ordering | None = None
    # The assumptions a dataset resubmit identity entry echoed back
    # (`ResubmitIdentityEntry.assumptions`). Its definite part states those that still hold
    # (`render.assumptions.assumptions_for`). None when the entry echoes none and on every
    # other door.
    assumption_echo: tuple[Assumption, ...] | None = None
    # How the part was counted (`ResolvedSpec.scope_choices`), set on a definite result
    # whose request posted scope entries or whose binders settled a scope axis. Its definite
    # part attaches it to its own Spec; a part folded into a same-shape merge does not.
    scope_choices: PartScope | None = None
    # The assumptions each reading of a dataset resubmit candidates entry echoed back
    # (`ResubmitCandidate.assumptions`), keyed by the rebuilt reading's spec_id. Its
    # candidates part reads each reading with its echo (`render.assumptions.assumptions_for`);
    # a reading absent here echoed none. Empty on every other door.
    reading_assumption_echoes: Mapping[str, tuple[Assumption, ...]] = field(default_factory=dict)
    # Unawaited coverage probe for streamed slot options, sent after `response`.
    # It leaves `specs[].slot_options` and the buffered response unchanged.
    # Its events and warnings feed the terminal `done` event.
    slot_options_task: asyncio.Task[_OptionProbeResult] | None = None
    # What this region's facet options are judged over: its StatVars, probe entities,
    # preview window and facet pins. Set on every definite result. A same-shape merge
    # builds the merged spec's probe from its members' and derives its options from it.
    facet_probe: FacetProbe | None = None
    # Relaunch a slot-options probe with a merged spec id and facet probe.
    # It captures the same shape, bindings, and graph clients as the original.
    # Only the event-loop thread may call it because it creates an asyncio task.
    slot_options_relaunch: (
        Callable[
            [str, list[SlotOptions], FacetProbe | None],
            asyncio.Task[_OptionProbeResult],
        ]
        | None
    ) = None
    # Unawaited observation fetch, streamed after the response with slot options.
    # Scheduled only with `stream_observations` and `include_observations`. Its
    # (events, warnings) result shares the slot-options await loop.
    observations_task: asyncio.Task[_ObsProbeResult] | None = None
    # Relaunch the fetch under a merged spec id. The observed StatVar/entity
    # pair stays unchanged. Reconciliation detects stale tasks by object identity.
    observations_relaunch: Callable[[str], asyncio.Task[_ObsProbeResult]] | None = None
    # Unawaited peers/children prefetch, streamed after the response. It needs
    # `stream_slot_options=True`, a materialise graph reader, and the finalized identity.
    # It is independent of slot-option coverage.
    # `[]` means the task ran without an observationAbout place. `None` marks
    # an unscheduled task.
    place_options_task: asyncio.Task[list[PlaceOptionsEvent]] | None = None
    # Launch for the region's current Spec, a merged one included; the task reads that
    # Spec's identity (`place_options.place_options_task`). The closure captures the same
    # graph and materialise clients.
    place_options_relaunch: Callable[[Spec], asyncio.Task[list[PlaceOptionsEvent]]] | None = None
    # What a resubmitted definite part's three relaunch closures read, apart from the
    # request's clients (`promote_relaunch.bind_relaunches`). The response cache's memo
    # stores the region without closures and binds them again for the request it serves.
    # None off the resubmit's definite path.
    promote_relaunch: PromoteRelaunch | None = None
    # What the definite answer's observation preview reads, apart from the request's
    # clients (`dispatch.PreviewRead`): the buffered read, the streamed task and the part
    # memo's rebound relaunch all read it. Set on a fresh or resubmitted definite answer and
    # a lone candidate survivor; None elsewhere, a same-shape merge included, whose preview
    # is a read of its merged identity (`same_shape._with_merged_preview`).
    preview_read: PreviewRead | None = None
    # Role-to-entity lists probed by resubmit resolution: the identity's own dcids on
    # each role. Populated on definite resubmit results.
    observation_entities: dict[str, list[str]] | None = None
    # Resubmit filters for source, method, and unit facets, applied before
    # facet selection. `None` means no bound axis or a non-resubmit result.
    facet_filter: Mapping[str, FacetAxisFilter] | None = None
    # Parsed inputs that would reproduce this region. `None` marks an unavailable
    # echo. Resubmit regions have no phrase inputs. Same-shape merges preserve it,
    # while part assembly reads restatement from each original region.
    restatement: ParsedPart | None = None
    # A resolved leg's inputs as one parsed part, set whether or not `restatement`
    # echoes them. `NoData.options` are built from it. The parsed doors set it to the
    # posted part. `None` when `ParsedPart` bounds reject the inputs, on soft-deadline
    # placeholders, and on resubmit identity, candidate and placeholder regions.
    parsed_form: ParsedPart | None = None
    # The leg's part is compared with the others (`ParsedPart.separate`), so the fold
    # (`same_shape.fold_key`) keeps it its own part.
    separate: bool = False
    # Operations the question asks for over this region's data, echoed onto its
    # part. Like restatement, part assembly reads them from each original region.
    operations: tuple[Operation, ...] = ()
    # The roles the shape that built the answering Spec declares, set where a definite or
    # candidates result is assembled: the definite Spec's shape, and the first candidate's
    # for a candidate set. `with_runnable_operations` reads it to tell which flow axes the
    # answer's data carries. None on every other result.
    answered_roles: tuple[str, ...] | None = None
    # Every candidate's `answered_roles`, in candidate order, on a candidate set the
    # pipeline assembles, so `every_member_phrases` judges every answer. None on every
    # other result.
    candidate_answered_roles: tuple[tuple[str, ...], ...] | None = None
    # Served measure partition and sibling partitions. Same-shape merges need
    # these to mint new identities after member materialisations are gone.
    measure_partition: MeasurePartition | None = None
    measure_siblings: tuple[MeasurePartition, ...] = ()
    # Open constraint axes and values asserted by served StatVars. Same-shape
    # merges combine all members' values and undeclared counts. Empty when no
    # definite member declares an open constraint.
    open_axis_facts: tuple[OpenAxisFact, ...] = ()
    # What this region's slot menus were counted from: per counted slot, the series a
    # binding of each option would serve (`option_coverage.option_axis_facts`).
    # A same-shape merge counts the merged menus from the union of these. Empty when no
    # menu was counted.
    option_axis_facts: tuple[OptionServedFact, ...] = ()
    # The in-memory served test this region's measure alternatives were counted with
    # (`option_coverage.measure_served_counter`), and the members its Spec serves and the
    # read plan whose rows it states, which its unit conversion offer reads
    # (`unit.conversion.with_unit_conversion`). A same-shape merge recounts the merged
    # alternatives over the union of the members' tests and offers conversions over the
    # union of their members and plans. None and empty off the definite path. The test is
    # also None for a subject-place family.
    served_filter: RoleServedFilter | None = None
    served_members: tuple[MemberFacts, ...] = ()
    served_plans: tuple[ReadPlan, ...] = ()
    # The form the Spec states its plan's coverage in (`retrieval.coverage.CoverageForm`),
    # which a same-shape merge states the merged plan's coverage in
    # (`same_shape._merged_counts`). None off the definite path and for a carrier that
    # keeps its own coverage.
    coverage_form: CoverageForm | None = None
    # Whether `served_plans` were narrowed to the units a unit request elected over this
    # answer's own group (`facets.FacetAdmission.elected_units`). A same-shape merge then
    # rebuilds no plan, since a resolve of the merged identity elects over the merged group.
    units_elected: bool = False
    # Each open directional role's member population behind this region's Spec, the
    # input its `ResolvedSpec.role_axis_facts` were counted from. A same-shape merge
    # counts the merged answer's members from the union of these, since the wire
    # facts carry counts and overlapping populations cannot be summed. Empty when no
    # role axis is open or the builder recorded none.
    role_axis_facts: tuple[RoleAxisFactInput, ...] = ()
    # Bindings-admitted population behind this region's StatVars. Same-shape
    # merges recompute the merged Spec's member count and digest from all
    # members because their materialisations are gone. `None` on non-definite
    # results or when population measurement was skipped.
    member_population: MemberPopulation | None = None
    # Whether the answer swallowed a read that failed or ran past its deadline and stands on
    # less than it would have read (`degraded_reads`): a member walk that settled nothing, a
    # named group dropped, the dates the unit gate could not read, the withheld-registry
    # re-read, a decision that consulted an unsettled default scope
    # (`place.ConsultedPresence`), or an alternate's presence read. Also set on the `no_data` placeholder of a leg whose own error the
    # fan-out swallowed, which reads as a real miss. Every door reports such a leg as degraded
    # (`DoneEvent.degraded_leg`), so neither cache layer stores the response, and the part
    # memo stores no such region.
    degraded: bool = False
    # The `time.monotonic()` instant this region's reads, its trailing streams included, run
    # to: the request deadline, or the later full-read bound when a resubmit's member walk
    # read the whole filtered set (`promote_reread._elected_members`). `None` when the region
    # was built without a deadline.
    reads_deadline: float | None = None
    # The departure facts raised beside this region's warnings, each under its warning's
    # key (`departures.FactsByWarning`). Part assembly keeps those whose warning the part
    # still holds, so a fact here whose warning was dropped reaches no part.
    departure_facts: FactsByWarning = field(default_factory=dict)
    # The request's own words for each name the answer states. They cover each entity name
    # extraction gave ("EU institutions" for "European Union"), where the raw-text door holds
    # them (`with_entity_words`), and each group the answer labels as the graph does, under
    # the name the request gave it (`place_roles.group_request_names`). The part's departures
    # and no-data frame quote a name in these words. A name absent here is quoted as given.
    # `None` on a door that holds no words for any entity (a resubmit), whose departures
    # then quote none.
    entity_words: Mapping[str, str] | None = field(default_factory=dict)
    # The words resolution read beside each entity name, where they differ from it
    # (`surface_resolution.resolve_names`): the query's words where they occur in it on the
    # raw-text door, a posted row's text beside its name on the parsed door. A choice posts
    # an ambiguous name's row with these words as its text, so the parsed door settles it
    # as this region did. A name absent here resolved as itself.
    entity_surfaces: Mapping[str, str] = field(default_factory=dict)
    # Whether the part's door asks for a flow side's members in the question's words ("by
    # donor"), as the raw-text door does (`raw_text._ExtractionPrep.onto`). The other doors
    # post `open_roles`. A side read whole states its way back to the members in that form.
    members_in_words: bool = False
    # The places the answer binds for each entity text of the part, by that text
    # (`place_roles.bound_entity_refs`), read once the part's places have settled. A text
    # absent here binds none: an ambiguous or unresolved one, a group or a set read as its
    # members in a part refused before its places are bound, and every text of a
    # soft-deadline placeholder.
    entity_refs: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # The flow sides a whole population the part asks for member by member opens
    # (`ResolveState.whole_listed`). The leg warns for each one the answer drops as it
    # does for its own open roles (`with_dropped_open_roles`).
    listed_axes: frozenset[RoleAxis] = frozenset()

    @property
    def spec(self) -> Spec:
        """Convenience for the definite path (status == 'definite')."""
        return self.specs[0]

    def words_of(self, name: str) -> str:
        """The request's own words for the entity `name` (`entity_words`), else `name`: what
        a refusal, option label or mention quotes for it."""
        return (self.entity_words or {}).get(name, name)

    @property
    def alternates_left_out(self) -> bool:
        """Whether the answer states `alternates_unevaluated`: an alternate was left
        unevaluated because its presence read did not complete."""
        return any(warning.code == ALTERNATES_UNEVALUATED for warning in self.warnings)


def any_degraded(regions: Iterable[RegionResult]) -> bool:
    """Whether any of `regions` stands on less than a later request may read, the
    `DoneEvent.degraded_leg` of every door."""
    return any(region.degraded for region in regions)


def source_refusal_result(
    variable_text: str, refusal: SourceRefusal, *, timing_by_step: Mapping[str, int]
) -> RegionResult:
    """The no-data result of a part whose source words resolved to no source or to several
    (`source_scope.SourceRefusal`), which runs no step of the pipeline."""
    return RegionResult(
        variable_text=variable_text,
        status="no_data",
        specs=(),
        no_data_reason=refusal.reason,
        warnings=(),
        timing_by_step=dict(timing_by_step),
        source_terms=refusal.terms,
        ambiguous_sources=refusal.ambiguous,
    )


def source_code_refusal_result(
    variable_text: str,
    refusal: SourceCodeRefusal,
    *,
    timing_by_step: Mapping[str, int],
    warnings: tuple[Warning, ...] = (),
    llm_usage: dict | None = None,
) -> RegionResult:
    """The no-data result of a part whose series code names no StatVar it can answer from
    (`source_code.SourceCodeRefusal`): the code is the outcome's source term, and an
    ambiguous code's sources are its candidates."""
    ambiguous = (
        (AmbiguousSource(text=refusal.code, candidates=refusal.sources),)
        if refusal.reason == "source_code_ambiguous"
        else ()
    )
    return RegionResult(
        variable_text=variable_text,
        status="no_data",
        specs=(),
        no_data_reason=refusal.reason,
        warnings=warnings,
        timing_by_step=dict(timing_by_step),
        llm_usage=llm_usage,
        source_terms=(refusal.code,),
        ambiguous_sources=ambiguous,
        source_code_refusal=refusal,
    )


def empty_scope_result(
    variable_text: str, scope: SourceScope, *, timing_by_step: Mapping[str, int]
) -> RegionResult:
    """The no-data result of a part whose source mentions allow no provenance, which runs
    no step of the pipeline: ``qualifier_not_bound`` under `scope`, so the outcome names
    the sources the part named, else those it excluded (`conjoin._unbound_terms`)."""
    return RegionResult(
        variable_text=variable_text,
        status="no_data",
        specs=(),
        no_data_reason="qualifier_not_bound",
        warnings=(),
        timing_by_step=dict(timing_by_step),
        source_scope=scope,
    )


def answered_roles(shape: ShapeDraft) -> tuple[str, ...]:
    """Return the roles a Spec built from `shape` answers under
    (`RegionResult.answered_roles`)."""
    return shape.declared_roles or ("observationAbout",)


def with_composite_answer(
    region: RegionResult,
    spec: Spec,
    *,
    observations_relaunch: Callable[[str], asyncio.Task[_ObsProbeResult]],
) -> RegionResult:
    """`region` answering `spec`, the composite its definite answer was restated as
    (`scope.composite.assemble_composite`). Its trailing slot options, place options and
    observation preview read the one table the region resolved, under that answer's spec_id,
    so the composite carries none of their tasks and no slot or place options. Its preview
    is read per component by `observations_relaunch` (`composite_answer.CompositeRelaunch`),
    which the caller launches on the event loop where the request streams observations
    (`enrichment._launch_trailing_tasks`).

    The restatement touches no task, so it runs on any thread. A task `region` held belongs
    to the request's event loop, and the caller cancels it there
    (`cancel_dropped_tasks`)."""
    return dataclasses.replace(
        region,
        specs=(spec, *region.specs[1:]),
        slot_options_task=None,
        slot_options_relaunch=None,
        place_options_task=None,
        place_options_relaunch=None,
        observations_task=None,
        observations_relaunch=observations_relaunch,
    )


def cancel_dropped_tasks(before: RegionResult, after: RegionResult) -> None:
    """Cancel each trailing task `before` holds that `after`, its restatement, does not
    (`with_composite_answer`). `Task.cancel` is not thread-safe, so the caller runs this on
    the event loop the tasks belong to."""
    held = (after.slot_options_task, after.place_options_task, after.observations_task)
    for task in (before.slot_options_task, before.place_options_task, before.observations_task):
        if task is not None and all(task is not kept for kept in held):
            task.cancel()


def with_runnable_operations(region: RegionResult, operations: Sequence[Operation]) -> RegionResult:
    """Return `region` carrying the `operations` its data can run, and a warning for each
    one it cannot.

    Every door stamps a part's operations through this function. An operation over a flow
    axis opens that axis (`fanout._stream_one_variable`), so a family with a by-creditor
    twin elects the twin, whose answer declares the role and keeps the operation, and its
    identity states the axis in `open_roles`. The operation is unrunnable on a role-bearing
    answer that declares no role on that axis (`RegionResult.answered_roles`): a
    recipient-only family holds one series per recipient, so a rank over creditors has
    nothing to rank. The operation leaves the part, and the part warns
    `query_qualifier_dropped`, recording a `breakdown_not_applied` departure under that
    warning. A subject-only answer, and a result that answers nothing,
    keep every operation.
    """
    roles = tuple(role for role in region.answered_roles or () if role != "observationAbout")
    if not roles:
        return dataclasses.replace(region, operations=tuple(operations))
    axes = {axis_for_role(role) for role in roles}
    unrunnable = [
        operation
        for operation in operations
        if operation.over in ("origin", "destination") and operation.over not in axes
    ]
    warnings = list(region.warnings)
    departure_facts = dict(region.departure_facts)
    for operation in unrunnable:
        warning = _unrunnable_warning(operation)
        # A parsed entry of a resubmit is stamped by the fan-out and again by its door.
        if warning not in warnings:
            warnings.append(warning)
        departure_facts[warning_key(warning)] = breakdown_not_applied(
            cast(RoleAxis, operation.over), asked=operation.phrase
        )
    return dataclasses.replace(
        region,
        operations=tuple(operation for operation in operations if operation not in unrunnable),
        warnings=tuple(warnings),
        departure_facts=departure_facts,
    )


def with_dropped_open_roles(region: RegionResult, open_axes: Collection[RoleAxis]) -> RegionResult:
    """Return `region` warning `query_qualifier_dropped` for each of `open_axes`, the part's
    own open roles and the sides a whole population it asks member by member opens
    (`RegionResult.listed_axes`), that its answer drops.

    A place the part names without `role_text` takes the axis place binding gives it, and a
    bound axis wins over an open one (`place_roles.PlaceRoles.open_axes`), so the answer
    holds that place's series where every member was asked for. A role-bearing answer that
    declares no role on an open axis (`RegionResult.answered_roles`) is the total over it,
    as for an operation over that axis (`with_runnable_operations`): the part warns and
    records a `breakdown_not_applied` departure, once per axis, so an axis a dropped
    operation already departs on raises nothing more. A subject-only answer, and a result
    that answers nothing, warn only for a bound axis. An axis every Spec binding it reads as
    a group's members (`_reads_members`) drops nothing, since the answer holds each member's
    series, so it warns nothing.
    """
    bound = {key for spec in region.specs for key in spec.definition.identity.entities}
    bound_alone = {
        key
        for spec in region.specs
        for key in spec.definition.identity.entities
        if not _reads_members(spec, key)
    }
    roles = tuple(role for role in region.answered_roles or () if role != "observationAbout")
    declared = {axis_for_role(role) for role in roles}
    departed = {
        fact.over
        for fact in region.departure_facts.values()
        if fact.code == "breakdown_not_applied"
    }
    warnings = list(region.warnings)
    departure_facts = dict(region.departure_facts)
    for axis in sorted(open_axes):
        if axis in bound_alone:
            warnings.append(_bound_open_role_warning(axis))
        elif axis in bound:
            continue
        elif roles and axis not in declared and axis not in departed:
            warning = _undeclared_open_role_warning(axis)
            warnings.append(warning)
            departure_facts[warning_key(warning)] = breakdown_not_applied(axis, asked=None)
    return dataclasses.replace(region, warnings=tuple(warnings), departure_facts=departure_facts)


def _reads_members(spec: Spec, role: str) -> bool:
    """Whether `spec` binds `role` as a named group's members (`EntityGroup`), one series per
    member. A group read as its published total or its whole population answers one series."""
    group = (spec.resolved.entity_groups or {}).get(role)
    return group is not None and group.total is None and not group.whole_of


def every_member_phrases(
    region: RegionResult, phrases: Sequence[str], *, query: str
) -> dict[str, RoleAxis]:
    """The `phrases`, stripped and casefolded, that ask for every member of a flow axis
    (`role_text.every_member_axis`, reading prepositions in `query`) and that `region`'s
    answer meets, each mapped to that axis.

    A role-bearing answer (`RegionResult.answered_roles`) that binds no entity on the axis
    holds every member on it: one series per member where it declares a role there, and
    the total over them where it declares none. An "all" phrase is met by either. An
    "each" or "every" phrase asks for the breakdown (`role_text.every_member_breakdown`),
    so only an answer declaring a role on the axis meets it, as for an open role
    (`with_dropped_open_roles`). A candidate set meets a phrase only when every candidate
    does (`RegionResult.candidate_answered_roles`). A subject-only answer and a result
    with no Spec meet none, so their phrases stay dropped.
    """
    if not region.specs:
        return {}
    answers = region.candidate_answered_roles or (region.answered_roles or (),)
    declared_by_answer = [
        {axis_for_role(role) for role in roles if role != "observationAbout"} for roles in answers
    ]
    if not all(declared_by_answer):
        return {}
    declared_by_all = set.intersection(*declared_by_answer)
    bound = {key for spec in region.specs for key in spec.definition.identity.entities}
    return {
        phrase.strip().casefold(): axis
        for phrase in phrases
        if (axis := every_member_axis(phrase, query=query)) is not None
        and axis not in bound
        and (axis in declared_by_all or not every_member_breakdown(phrase))
    }


def _bound_open_role_warning(axis: RoleAxis) -> Warning:
    """The `query_qualifier_dropped` warning for an open role a named place binds."""
    noun = axis_noun(axis)
    return Warning(
        code=QUERY_QUALIFIER_DROPPED,
        severity="degraded",
        message=(
            f"The requested breakdown over every {noun} was not applied: the part names "
            f"a {noun}, and the answer is for that {noun}."
        ),
    )


def _undeclared_open_role_warning(axis: RoleAxis) -> Warning:
    """The `query_qualifier_dropped` warning for an open role the answer declares no role
    on."""
    noun = axis_noun(axis)
    return Warning(
        code=QUERY_QUALIFIER_DROPPED,
        severity="degraded",
        message=(
            f"The requested breakdown over every {noun} was not applied: the answer is the "
            f"total over every {noun}, and a breakdown by {noun} needs a named {noun}."
        ),
    )


def with_departure_facts(region: RegionResult, facts: FactsByWarning) -> RegionResult:
    """Return `region` also holding `facts`, the departures raised beside warnings that
    reach its part from outside it, such as the envelope's. Where both raised a fact beside
    one warning, the region's own fact wins, since it was raised beside the binding it names."""
    if not facts:
        return region
    return dataclasses.replace(region, departure_facts={**facts, **region.departure_facts})


def _unrunnable_warning(operation: Operation) -> Warning:
    """The `query_qualifier_dropped` warning for an operation the answer cannot run."""
    noun = axis_noun(cast(RoleAxis, operation.over))
    return Warning(
        code=QUERY_QUALIFIER_DROPPED,
        severity="degraded",
        message=(
            f'The requested "{operation.phrase}" was not applied: the answer is the total '
            f"over every {noun}, and a breakdown by {noun} needs a named {noun}."
        ),
    )


def with_entity_words(
    region: RegionResult,
    words: Mapping[str, str],
    *,
    surfaces: Mapping[str, str] = NO_ENTITY_WORDS,
) -> RegionResult:
    """Return `region` holding `words`, the request's own words for each entity name
    (`RegionResult.entity_words`), and `surfaces`, the words resolution read beside each name
    (`RegionResult.entity_surfaces`). A group label `region` already maps to the request's
    name for the group maps on to that name's words."""
    if not words and not surfaces:
        return region
    by_label = {label: words.get(name, name) for label, name in (region.entity_words or {}).items()}
    return dataclasses.replace(
        region,
        entity_words={**words, **by_label},
        entity_surfaces={**region.entity_surfaces, **surfaces},
    )
