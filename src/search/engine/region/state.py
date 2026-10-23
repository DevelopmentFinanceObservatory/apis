"""Per-call state shared by the Search resolve stages.

Request context is fixed at construction; pipeline state accumulates as stages
run. Both `region.pipeline` and `region.candidates` import the type here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, cast

from code_index.members import FamilyQuery
from code_index.shared import get_index
from code_index.types import BroadReading
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from dfo_platform import Cause
from entity_resolution import Admission, EntityResolver, Resolution
from materialise import GraphReader
from materialise.facets import FacetSelection
from materialise.place_scope import DefaultPresence
from materialise.rate_tables import RateTableCache
from origin_catalogue import UnUnserved
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.binders import Binder, BindResult, GeminiBinder
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.departures import (
    WarningKey,
    measure_not_applied,
    warning_key,
    window_gap_fact,
)
from search.engine.errors import GraphInfraError
from search.engine.events import EventSink
from search.engine.extract import DateRequest
from search.engine.frame import Frame, door_frame
from search.engine.interpret import PinnedResolved, Recall
from search.engine.interpretation import Interpretation, InterpretationSet
from search.engine.place import read_default_presence
from search.engine.qualifier_match import (
    CodedQualifier,
    QualifierMatch,
    QualifierReadings,
    QualifierSpan,
    QualifierTie,
    _normalize,
)
from search.engine.quantity import (
    DenominatorKind,
    DenominatorRead,
    NamedDenominator,
    NamedNormalisation,
    QuantityRequirement,
    quantity_requirement,
)
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from search.engine.region.convertibility import UnitCoverage
from search.engine.region.materialise_step import Departed, family_total_absent_departures
from search.engine.region.observations.entities import open_role_axis
from search.engine.region.place_context import (
    PlaceContext,
    UnsettledPlace,
    default_scope_stat_vars,
    empty_place_context,
)
from search.engine.region.place_roles import PlaceAssignment, place_order, place_texts
from search.engine.region.place_set import PlaceSet
from search.engine.region.result import (
    AmbiguousEntity,
    FacetAxisFilter,
    RegionResult,
    RoleNotAdmittedEntity,
    source_code_refusal_result,
)
from search.engine.region.status_rule import CLOSED_AXES, OpenAxes, StatusPlan
from search.engine.retrieval.bound_values import (
    MeasurePartition,
    RoleClassWalks,
    RoleKeys,
    WalkedMembers,
    held_codes,
    identity_query,
    index_decides_membership,
    query_bound_values,
    query_members,
    shape_member_facts,
    shape_population_arcs,
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
from search.engine.retrieval.standard import BreakdownDecision, Keep, default_scope_members
from search.engine.role_text import DoorRole
from search.engine.shape import Pool, ShapeDraft, is_role_bearing, role_names_by_axis
from search.engine.source_code import SeriesCodeMembers, not_held
from search.engine.source_scope import SourceScope
from search.engine.unit.bind import BoundUnitRequest, RequestBasis
from search.heading.departure_fact import DepartureFact
from search.models import (
    BaseIdentity,
    CandidateBasis,
    Coverage,
    Entity,
    EntityRoleKey,
    EntryPath,
    FacetBasis,
    GraphRef,
    Headline,
    IdentityUnitRequest,
    NearestRealSpec,
    NoDataDenominator,
    NoDataGate,
    NoDataReason,
    Operation,
    ProvenanceRef,
    ResolvedFacet,
    ResolvedQualifier,
    RoleAxis,
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

    # `requirements` imports this module.
    from search.engine.region.requirements import Judgement

__all__ = ["EngineOptions", "ResolveState", "UnLeg", "raised_facts"]


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

    Only role-incompatible terminals may opt into the ShapeDraft's
    confirmed-member fallback. Labels use each member's `->name` arc among the members the
    shape stands for (`shape_population_arcs`), falling back to its dcid.
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
    arc_facts = shape_population_arcs(shape_draft)
    return [
        GraphRef(ref=dcid, label=read_name(arc_facts.get(dcid, {})) or dcid) for dcid in sv_dcids
    ]


def served_stat_vars(
    shape: ShapeDraft, bindings: Sequence[SlotBindingDraft], role_keys: RoleKeys
) -> list[str]:
    """The StatVars a Datasets mint serves for the Spec `shape` grounds under `bindings` with
    its places bound under `role_keys`, in member order.

    A subject-only shape's are the members `standard_resolve` reads at their own default
    place scope (`standard.default_scope_members`), the family the mint keeps and the
    members the resolve answers through. A role-bearing shape's are `_served_members` over
    `shape_query`."""
    if not is_role_bearing(shape):
        return default_scope_members(shape, list(bindings), role_keys.bound)
    return _served_members(
        shape, shape_query(shape, bindings, role_keys.bound, open_role_keys=role_keys.open)
    )


@dataclass(frozen=True)
class TakerStatVars:
    """The StatVars of a shape taking a default scope under its bindings: those a mint serves
    for it (`served_stat_vars`), which its place context takes the scopes of, and those its
    presence read covers (`presence_stat_vars`)."""

    served: tuple[str, ...]
    covered: tuple[str, ...]


def taker_stat_vars(
    shape: ShapeDraft, bindings: Sequence[SlotBindingDraft], role_keys: RoleKeys
) -> TakerStatVars:
    """`shape`'s served and covered StatVars under `bindings` and `role_keys`, electing its
    served members once for both. For a shape standing for a held family's members that
    election reads the whole population, so the phases make it off the event loop."""
    served = tuple(served_stat_vars(shape, bindings, role_keys))
    return TakerStatVars(served, tuple(_covered(shape, served)))


def presence_stat_vars(
    shape: ShapeDraft, bindings: Sequence[SlotBindingDraft], role_keys: RoleKeys
) -> list[str]:
    """The StatVars a default-scope presence read covers for `shape` answered under
    `bindings` with its places bound under `role_keys`. Every reader of that presence for
    the shape reads within them: the read itself (`phase_place_materialise._group_stat_vars`),
    the facet prewarm (`candidates._prewarm_places`), the place context and the resolve.

    They are the members the shape holds when those are its recall (`default_scope_stat_vars`,
    which refuses an index-sourced shape), then, for a shape that stands for members past
    its recall, the members a mint serves for it under `bindings` (`served_stat_vars`): a
    shape holding a catalogue family (`ShapeDraft.held_family`) and one narrowed from a
    family's members (`ShapeDraft.walk_seeds`), whose own members are not read. Those are
    the members `standard_resolve` reads at their own scope under `bindings`, so a reader
    that changes the bindings, as unbinding does, tops the presence up over this function
    for the new ones. A role-bearing shape takes no default scope and covers none."""
    if is_role_bearing(shape):
        return []
    past_recall = shape.held_family is not None or shape.walk_seeds is not None
    return _covered(shape, served_stat_vars(shape, bindings, role_keys) if past_recall else ())


def _covered(shape: ShapeDraft, served: Sequence[str]) -> list[str]:
    """What `presence_stat_vars` returns for `shape`, given the StatVars a mint serves for it
    under its bindings."""
    if is_role_bearing(shape):
        return []
    recalled = default_scope_stat_vars([shape]) if shape.walk_seeds is None else []
    past_recall = shape.held_family is not None or shape.walk_seeds is not None
    return list(dict.fromkeys([*recalled, *(served if past_recall else ())]))


def presence_lacking(
    presence: DefaultPresence,
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    role_keys: RoleKeys,
) -> list[str]:
    """The StatVars a resolve of `shape` under `bindings` reads at their own default scope
    (`presence_stat_vars`) that `presence` does not cover."""
    return [
        sv for sv in presence_stat_vars(shape, bindings, role_keys) if sv not in presence.scopes
    ]


def presence_covering(
    presence: DefaultPresence,
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    role_keys: RoleKeys,
    *,
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    deadline: float | None,
) -> DefaultPresence:
    """`presence` topped up, in one presence read (`place.read_default_presence`), with the
    StatVars a resolve of `shape` under `bindings` reads at their own default scope that it
    lacks (`presence_stat_vars`), for a reader that resolves `shape` under bindings the read
    was not made for, as unbinding values that match no member together does. `presence`
    itself when it covers them. Raises what the presence read raises."""
    missing = presence_lacking(presence, shape, bindings, role_keys)
    if not missing:
        return presence
    return read_default_presence(
        graph=graph, sdmx=sdmx, stat_vars=missing, known=presence, deadline=deadline
    )


def identity_served_stat_vars(identity: BaseIdentity, shape: ShapeDraft) -> list[str]:
    """The StatVars a Datasets mint serves for a resubmitted `identity`, in member order.

    `shape` holds its members exactly (`ShapeDraft.sv_arc_facts_complete`): the family the
    mint's walk resolves for the identity (`elected_role_class_shape`), a walk that takes
    no recalled members as seeds (`ResubmitWalk`). The mint resolves an identity the code
    index decides through the index (`index_decides_membership`), so for that identity the
    index's family answers."""
    if index_decides_membership(identity):
        shape = replace(shape, sv_arc_facts_complete=False)
    return _served_members(shape, identity_query(identity))


def _served_members(shape: ShapeDraft, query: FamilyQuery) -> list[str]:
    """The StatVars a Datasets mint serves for `shape` under `query`.

    The population is the one `shape_member_facts` resolves. For a query the code index
    decides, that is the elected dataset's family (`code_index.members.index_family`), and
    the mint serves its antichain-minimal rows. For a shape the index does not decide, it is
    the walked family once the mint's walk settles the members, which the shape then holds
    exactly (`elected_role_class_shape`), and the members the shape holds otherwise, which
    the answer flags `membership_unverified`. A shape that stands for a family the catalogue
    holds (`ShapeDraft.held_family`) holds members outside the query's bound values, so its
    population is the one `query_members` confirms for the query's slots, as the walk
    would. A role-bearing population is reduced to the members the answer lists
    (`role_resolve.role_members_served`), and any other to its antichain-minimal members."""
    read = query_members if shape.held_family is not None else shape_member_facts
    member_facts, index_sourced = read(shape, query)
    if is_role_bearing(shape) and not index_sourced:
        return role_members_served(
            member_facts, query_bound_values(query), query.role_keys, query.open_role_keys
        )
    return minimal_constraint_svs(
        list(member_facts), {sv: member.read_constraints() for sv, member in member_facts.items()}
    )


# Why the presence read an alternate's default scope needs did not complete: no worker of
# the alternates' presence pool was free, the presence source failed, or it timed out.
AlternateUnevaluatedReason = Literal["no_capacity", "upstream_error", "timed_out"]


def shape_vocabulary(shape: ShapeDraft) -> frozenset[str]:
    """The constraint properties `shape` can carry into binding, its slots' properties, which
    are also the code systems it reads qualifier text in. A slot a breakdown decision left out
    (`ShapeDraft.omitted_slots`) stays in it: the query was read in the family's vocabulary,
    and a term on that slot is one the answer leaves unapplied."""
    return (
        frozenset(slot.property_dcid for slot in shape.slot_keys if slot.property_dcid)
        | shape.omitted_slots
    )


def qualifier_dropped_warning(phrase: str) -> Warning:
    """The `query_qualifier_dropped` warning for a qualifier phrase the answer does not
    apply."""
    return Warning(
        code=QUERY_QUALIFIER_DROPPED,
        severity="degraded",
        message=f'The requested qualifier "{phrase}" was not applied.',
    )


@dataclass(frozen=True)
class CodeRule:
    """An answer meets the rule when its bindings bind one of `codes`, or a descendant
    (`CodeIndex.expansion`), on `property_dcid`, and leaves it unmet when its shape's slots
    hold the property (`shape_vocabulary`) and its bindings bind none of them. An answer
    whose shape lacks the property is outside it: the code is the model's judgement of the
    measure."""

    property_dcid: str
    codes: tuple[str, ...]


@dataclass(frozen=True)
class ValueRule:
    """The values a run of the request's words names by its graph label, as (property,
    value) pairs, and the code systems of the code terms the words overlap ("primary
    education" is the text of CRS purpose 11220). An answer applies the run as
    `phase_rank_bind.judge_value_requirements` decides, and an answer whose shape holds
    one of `code_systems` leaves the words to its code readers."""

    values: frozenset[tuple[str, str]]
    code_systems: frozenset[str] = frozenset()


@dataclass(frozen=True)
class MeasureRule:
    """A measure value the request's words name (`match`) that the elected shape does not
    carry under the bindings it was elected with. An answer meets the rule when its shape
    carries the value or a value its settled bindings bind reads it
    (`qualifier_readings.answer_reads_measure`), which binding repair can bind after
    election: activitySource GrossDomesticProduction reads "gross". Otherwise it is judged
    as `code`, the `CodeRule` on the measure axis."""

    match: QualifierMatch
    code: CodeRule


# Where a requirement was recorded: a model tie the bindings leave unresolved ("tie"), the
# elected reading's own measure phrases, coverage words and query records ("bind"), a run
# naming a value by its graph label ("value"), a value the bind drafted on a slot the
# elected reading lacks ("stray"), each value of such a set of two or more ("set_value"), a
# request-named value unbinding drops ("unbinding"), or a value a resubmit posts and cannot
# apply ("posted").
type RequirementOrigin = Literal[
    "tie", "bind", "value", "stray", "set_value", "unbinding", "posted"
]

# How an answer is judged on a requirement: by its codes (`CodeRule`), by the measure value
# its words name (`MeasureRule`), by the values its words name (`ValueRule`), as unmet on
# every answer ("unbound", a query record of the elected reading no binding covers), or on none (`None`, a term no one slot clears, which
# departs unless a refusal names it).
type RequirementRule = CodeRule | MeasureRule | ValueRule | Literal["unbound"] | None


@dataclass(frozen=True)
class Requirement:
    """A term the request names that every final answer is judged against
    (`requirements.judge`), recorded once in the part's ledger (`ResolveState.requirements`).
    Nothing departs when it is recorded: the judgement of the final answer decides whether
    `departure` is emitted (`requirements.with_requirement_departures`).

    `term` is what a trace and a `qualifier_not_bound` refusal name when an answer leaves it
    unmet, and its normalised text is the requirement's `phrase`. `departure` is the
    warning and `measure_not_applied` fact that disclose it when no refusal names it,
    `None` when it names nothing (a value no name exists for). A departure several
    requirements carry discloses each of their phrases; `alone` then discloses this one's
    phrase without the others, for when some of them need no disclosure. `generic` is a
    departure naming no phrase in particular ("A requested gender value was not applied."),
    true of any of its requirements left unapplied, so it discloses those of its phrases
    that need it whatever the others. `refuses` is `False` for a value only the bind model
    read (`phase_rank_bind.model_only_values`). An answer is judged on it, and one leaving
    it unapplied discloses it and stands."""

    term: ResolvedQualifier
    origin: RequirementOrigin
    rule: RequirementRule = None
    departure: Departed | None = None
    alone: Departed | None = None
    generic: Departed | None = None
    refuses: bool = True

    @property
    def phrase(self) -> str:
        return _normalize(self.term.text)


def code_requirement(
    *,
    property_dcid: str,
    codes: tuple[str, ...],
    asked: str | None,
    named: Sequence[GraphRef],
    departure: Departed | None,
    origin: RequirementOrigin,
    alone: Departed | None = None,
    generic: Departed | None = None,
    refuses: bool = True,
) -> Requirement:
    """A requirement that an answer apply `codes` on `property_dcid` (`CodeRule`), named by
    `asked`, the request's words, and `named`, the code rows they read as. A value no name
    exists for (`asked` `None`) is named by its dcid. `departure`, `alone`, `generic` and
    `refuses` are the `Requirement`'s."""
    if asked is None:
        term = ResolvedQualifier(
            text=" and ".join(codes),
            named=[GraphRef(ref=code, label=code) for code in codes],
            slot=property_dcid,
            disposition="unbound",
        )
    else:
        term = ResolvedQualifier(
            text=asked, named=list(named), slot=property_dcid, disposition="unbound"
        )
    return Requirement(
        term=term,
        origin=origin,
        rule=CodeRule(property_dcid=property_dcid, codes=codes),
        departure=departure,
        alone=alone,
        generic=generic,
        refuses=refuses,
    )


def unapplied_departure(phrase: str, *, asked: str, property_dcid: str | None) -> Departed:
    """The `query_qualifier_dropped` warning for `phrase` and the `measure_not_applied` fact
    it discloses for the request's words `asked`, cleared by the slot of `property_dcid`."""
    return (
        qualifier_dropped_warning(phrase),
        measure_not_applied(asked=asked, property_dcid=property_dcid),
    )


def unjudged_requirement(
    phrase: str, *, asked: str, property_dcid: str | None, origin: RequirementOrigin
) -> Requirement:
    """A term no answer is judged on (rule `None`): it departs `measure_not_applied` for
    the request's words `asked`, cleared by the slot of `property_dcid`, unless a refusal
    names `phrase`."""
    return Requirement(
        term=ResolvedQualifier(text=asked, named=[], disposition="unbound"),
        origin=origin,
        departure=unapplied_departure(phrase, asked=asked, property_dcid=property_dcid),
    )


def with_requirement(
    requirements: tuple[Requirement, ...], requirement: Requirement
) -> tuple[Requirement, ...]:
    """`requirements` with `requirement` added, once: a phase that runs again over the
    same state records the same requirement again."""
    return requirements if requirement in requirements else (*requirements, requirement)


def for_shape[T](
    by_vocabulary: Mapping[frozenset[str], tuple[T, ...]], shape: ShapeDraft
) -> tuple[T, ...]:
    """`shape`'s own entry of a map keyed by vocabulary (`shape_vocabulary`), empty when the
    query was not read in its vocabulary."""
    return by_vocabulary.get(shape_vocabulary(shape), ())


def merge_lists[T](lists: Iterable[Iterable[T]]) -> tuple[T, ...]:
    """The union of `lists` in first-seen order, each item as often as the one list that
    repeats it most. Two vocabularies' lists of one query share their common matches, and a
    match the variable and the query text both name appears twice in each, so the union
    reads as one list would."""
    merged: list[T] = []
    for items in lists:
        seen: dict[T, int] = {}
        for item in items:
            seen[item] = seen.get(item, 0) + 1
            if seen[item] > merged.count(item):
                merged.append(item)
    return tuple(merged)


@dataclass(frozen=True, slots=True)
class UnPool:
    """What the UN leg gives (`phase_intake._un_shapes`): the UN pool's shapes, why the part
    has none to fill a gap with (`None` when it need not), and what the leg's own reads could
    not serve, the UN places they named that the UN front door refuses and their failed UN
    reads (`unserved`), and the reads they swallowed (`lost`). Those reads describe the UN
    pool alone, so they join the part's records only when the pool reaches a menu: on the
    UN pass (`pipeline._fill_gap`), or merged onto ours (`phase_intake._merge_un_pool`)."""

    shapes: list[ShapeDraft]
    missed: str | None = None
    unserved: UnUnserved = field(default_factory=UnUnserved)
    lost: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UnLeg:
    """The UN System Data Commons' resolve and the derivation of its shapes, running as one
    task beside our pass (`phase_intake._un_shapes`), and the monotonic time the wait for
    it ends: `SEARCH_UN_FIRST_PASS_WAIT_S` past the start of our derivation."""

    task: asyncio.Task[UnPool]
    ends: float


@dataclass(frozen=True, slots=True)
class SlotDefaultDisclosure:
    """What one declared default that fired states (`phase_election._repair_bindings`): its
    `slot_defaulted` warnings, the normalised dropped phrases it answers, whose
    entry-level warning those replace, and the broad reading it is, if any
    (`bind_repair.apply_broad_readings`), whose broader value applies the words of its
    narrow one (`phase_rank_bind.judge_value_requirements`)."""

    warnings: tuple[Warning, ...] = ()
    phrases: frozenset[str] = frozenset()
    reading: BroadReading | None = None


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
    # The entity names the door asks for as their members, each member its own series.
    # Place resolution gives each one's set the members reading (`PlaceSet.reading`).
    member_entities: frozenset[str] = frozenset()
    # Extracted dropped phrases shared across every variable leg, followed on the
    # raw-text door by this leg's own operation phrases; each binder checks them
    # against its shape, then warnings combine admission across regions.
    dropped_qualifier_phrases: list[str] = field(default_factory=list)
    # This leg's admitted operations. One over `category` opens the slots it names on
    # the elected shape (`phase_rank_bind._open_operation_slots`).
    operations: tuple[Operation, ...] = ()
    # The flow axes this leg asks every member of. `bind_place_roles` leaves open those no
    # place binds (`PlaceRoles.open_axes`).
    open_axes: frozenset[RoleAxis] = frozenset()
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
    # The binder the part's quantity read and bind call go through, on `llm`.
    binder: Binder = field(default_factory=GeminiBinder)
    sdmx: EngineSdmxClient
    deadline: float | None
    pre_resolved: dict[str, Resolution] | None
    # The query words naming an entity, which recall resolves it through
    # (`surface_resolution.resolve_names`). The raw-text door alone supplies them.
    entity_surfaces: dict[str, str] = field(default_factory=dict)
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
    # The door the leg was posted on, which decides who states each frame field
    # (`frame.door_frame`). The dataset_resubmit door's parsed entries resolve as parsed.
    door: EntryPath = "parsed"
    # The recall pool the pass elects over: our candidates, or the UN System Data Commons'
    # on the pass that runs when ours ends no_data (`pipeline._fill_gap`).
    pool: Pool = "ours"

    # What the part asks (`frame.Frame`): its door-time fields from construction, completed
    # with what the offered readings' vocabularies read in its words once the part seats a
    # reading or refuses on a source exclusion (`phase_rank_bind._hold_offer`), or with the
    # refused readings' when it offers none (`phase_rank_bind._refused_menu`).
    frame: Frame = field(init=False)

    # --- (b) evolving pipeline state ------------------------------------------
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
    # What the mint states of the members the walk settled for the shape in play, `None`
    # when no walk settled them (`MembershipResolution.walked`).
    walked: WalkedMembers | None = None
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
    # Per-StatVar unit election from `phase_answer._ground_elected_answer`; later scope and
    # observation reads reuse it to avoid repeating the graph/SDMX lookup.
    unit_elected_selections: Mapping[str, tuple[Mapping[str, FacetSelection], ...]] | None = None
    # The per-variable recall() result: candidate stat-vars and the
    # Country-level entity names recall() alone resolved. None before recall runs.
    rcl: Recall | None = None
    # Pre-expansion Country dcids from recall; derive_shapes uses this original set.
    recall_only_place_dcids: list[str] | None = None
    # Each place the request names that resolved, by the request's text for it, with groups
    # first, then places and regions, each in recall's order. Set by `_phase_resolve_places`. A name
    # with no entry did not resolve.
    place_sets: dict[str, PlaceSet] = field(default_factory=dict)
    # The place set of the party each word of the part names on an axis the request leaves
    # unnamed (`place_roles.stated_party_words`), by the party's dcid. Set by
    # `_phase_resolve_places`; `place_roles.bind_place_roles` binds one on a shape whose
    # datasets declare the word.
    party_sets: dict[str, PlaceSet] = field(default_factory=dict)
    # Whether the part is read without any word naming a party
    # (`place_roles.stated_party_words` then returns none): the reading
    # `pipeline.resolve_variable` falls back to when the party reading binds no party, and
    # the reading of every UN pass.
    party_words_off: bool = False
    # Start tick for the shape timing span, including derivation and ranking.
    shape_t0: int = 0
    # All confirmed shapes, before eligibility and ranking filters.
    shapes_found: list[ShapeDraft] = field(default_factory=list)
    # Whether a role-bearing eligible shape outranks the subject-only ones
    # (`phase_rank_bind._role_bearing_outranks`), set by `_phase_prune_and_rank`.
    role_bearing_outranks: bool = False
    # The normalisation this leg names, read from its variable and its sidecar qualifier
    # phrases (`quantity.quantity_requirement`), with the form the quantity read states
    # joined before admissibility when no phrase named one (`quantity.with_stated_form`).
    quantity_requirement: QuantityRequirement = field(init=False)
    # The kind and name of each StatVar denominator the shapes admitted on their measure
    # and roles divide by, read in one batch before the form rule
    # (`phase_rank_bind._read_denominators`). The quantity read lists them
    # (`quantity_read.read_stated_quantity`).
    denominator_reads: dict[str, DenominatorRead] = field(default_factory=dict)
    # The error a failed denominator read raised on a leg that named no form, held so the
    # read fails the part once the quantity read states a form or names a denominator
    # (`phase_rank_bind._read_quantity_answer`). `None` when the read succeeded.
    denominator_read_failure: GraphInfraError | None = None
    # The kinds of those denominators, held once the requirement names a form, from the
    # leg's phrases or the form the quantity read states. Admissibility, materialise and
    # the alternate guard read them.
    denominator_kinds: dict[str, DenominatorKind] = field(default_factory=dict)
    # The form the quantity read stated when no phrase named one, joined to
    # `quantity_requirement` and recorded on the trace as model-stated. `None` otherwise.
    stated_form: NamedNormalisation | None = None
    # The denominator the question names, as the quantity read chose it from `denominator_reads`
    # (`quantity.NamedDenominator`). `None` when the question names none. Materialise
    # holds the elected shape to it (`quantity.check_denominator`).
    named_denominator: NamedDenominator | None = None
    # The admissible shapes the narrowing leaves before the measure-family collapse, less
    # those a source exclusion removes; written once the menu seats one of them or the
    # exclusion leaves none.
    eligible_shapes: list[ShapeDraft] = field(default_factory=list)
    # The admissible shapes the menu is seated from, uncut: in rank order (recall or
    # topic-root score, then breadth and shape id), with measure-family siblings of the
    # indexed default collapsed and a part's excluded sources out. No engine code reads
    # it after the phase that writes it; tests and diagnostics do.
    ranked_shapes: list[ShapeDraft] = field(default_factory=list)
    # The seated bind menu, at most `bind_top_k` shapes; `chosen_shape_index` indexes it.
    capped_shapes: list[ShapeDraft] = field(default_factory=list)
    # Every reading of the part, admitted or refused, with the admissible ones the bind
    # menu shows and the ones refused only on a pinned part (`region.admissibility`).
    interpretations: InterpretationSet = field(default_factory=InterpretationSet)
    # The reading refused on a pinned part that the bind model named as nearest when it
    # declined every admissible one. It stands in as the elected shape so the refusal can
    # flag it with its bindings and places settled, and the part refuses before
    # materialise (`phase_place_materialise._declined_refusal`). `None` otherwise.
    declined: Interpretation | None = None
    # Recognised qualifiers from the variable, query text, dropped phrases and the typed
    # sidecar, one list per vocabulary (`shape_vocabulary`) among the eligible shapes, each
    # read in that vocabulary's code systems alone. Computed once before measure collapse
    # and reused for narrowing and admission; `for_shape` reads one shape's list and
    # `all_qualifier_matches` their union.
    query_qualifier_matches: dict[frozenset[str], tuple[QualifierMatch, ...]] = field(
        default_factory=dict
    )
    # The query's code matches in the vocabulary of every reading judged on the pinned parts
    # (`admissibility.judge_pinned_parts`), those refused on one included.
    pinned_part_matches: dict[frozenset[str], tuple[QualifierMatch, ...]] = field(
        default_factory=dict
    )
    # Every recognised span in `variable`, including ambiguities, as each candidate
    # shape's vocabulary reads it; the elected shape's builds `ResolvedQualifier.named`
    # (`for_shape`). Raw `query_text` is excluded: span recognition accepts ambiguous terms
    # that admission rejects, which would trigger a false `qualifier_not_bound` refusal.
    # Query-text matches still reach admission.
    variable_spans_by_vocabulary: dict[frozenset[str], tuple[QualifierSpan, ...]] = field(
        default_factory=dict
    )
    # Every recognised span in `variable` read in every code system together: the code
    # terms the named-dataset check refuses and the elected shape's unread words warn on.
    variable_spans_in_every_system: tuple[QualifierSpan, ...] = ()
    # Normalised phrases matched in `variable` or `query_text`, plus tie phrases.
    # Excludes dropped-only phrases, whose warning ownership belongs to raw-text entry.
    variable_matched_qualifier_phrases: frozenset[str] = frozenset()
    # The provenances the part may answer from, compiled from its source mentions
    # (`search.engine.source_scope`), or `None` when it names and excludes none. Shape
    # candidacy (`phase_rank_bind._dataset_candidates`) and every election read it, and the
    # read applies it through the bound catalogue (`pipeline.resolve_variable`).
    scope: SourceScope | None = None
    # The StatVars the part's series code names (`search.engine.source_code`), which recall
    # reads in place of the embedding resolve (`interpret.recall`), or `None` when the part
    # names no code.
    source_code: SeriesCodeMembers | None = None
    # Whether the part's scope removed a reading where it applies: candidacy for a part that
    # names a source, and for one that only excludes a reading the menu would have offered
    # without the exclusion and does not offer with it
    # (`phase_rank_bind._phase_prune_and_rank`). A part left with no reading it accepts
    # refuses naming its sources only then (`phase_rank_bind._declined_terminal`).
    scope_removed_readings: bool = False
    # Whether the part, which only excludes sources, refused `qualifier_not_bound` on its
    # exclusion (`phase_rank_bind.exclusion_records`, the one place that sets it): the
    # exclusion left no reading that seats, the model declined what it left, or the scoped
    # read found the elected reading's data excluded at the places. Our catalogue holds the
    # measure and the part left its publishers out, so the UN pass, which fills gaps in our
    # catalogue, does not run (`pipeline._un_pass_skip`). A part naming a source never sets
    # it, and nor does a refusal of the election for codes or rows the part's tables lack
    # (`phase_election._vet_election`, `phase_election._unserved_sources`).
    scope_settled: bool = False
    # The UN System Data Commons' resolve and shape derivation, started beside our pool's
    # derivation (`phase_intake._phase_derive_shapes`), `None` on the UN pass and on a part
    # whose scope allows no UN provenance or that ended before derivation.
    un_leg: UnLeg | None = None
    # The UN leg's pool when our pass took it under `SEARCH_UN_ON_FIRST_MENU`
    # (`phase_intake._merge_un_pool`): merged onto its menu, or late or failed after the
    # pass waited for it, which `pipeline._fill_gap` warns on. `None` when the pass left the
    # leg to the UN pass.
    un_pool: UnPool | None = None
    # Whether the part posted source mentions, so each answer's identity binds the
    # provenances that served it (`spec_build.build_spec`'s `pin_served_provenance`).
    pin_served_provenance: bool = False
    # Ambiguous qualifier matches across variable, single-variable query text, dropped
    # phrases and the sidecar, one list per vocabulary as `query_qualifier_matches`. Bind
    # narrows them to one value or alternatives.
    query_qualifier_ties: dict[frozenset[str], tuple[QualifierTie, ...]] = field(
        default_factory=dict
    )
    # Spans naming one code row and one measure value, from the same sources as the ties.
    # Each reading is admitted only against a shape that carries it alone.
    query_qualifier_readings: tuple[QualifierReadings, ...] = ()
    # Each shape's held-value reading (`bound_values.held_codes`), built once per shape for
    # the family election, the axes rule and tie admission (`held_codes_of`). The shape
    # stays beside its reading, so its id names that shape for the whole resolve.
    shape_held_codes: dict[int, tuple[ShapeDraft, Callable[[QualifierMatch], bool]]] = field(
        default_factory=dict, repr=False
    )
    # Whether a dropped qualifier phrase uses a price-basis word no match read
    # (`alternate_guard.leaves_price_basis_unread`). The default-reading collapse and the
    # guard's default operation both stand down on such a leg.
    unread_price_basis: bool = False
    # What the default-reading collapse did before bind, which every Spec's trace states.
    default_reading: TraceDefaultReading | None = None
    # Ties with multiple rows after declared-property narrowing, admitted against the capped
    # shapes and then narrowed to the elected shape's own; bind shows each shape its own as
    # alternatives, and qualifier guards treat them as attempted matches. A single surviving
    # row becomes an admitted qualifier instead.
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
    # The codes the bind model chose for the elected shape from a phrase's candidate menu
    # that read that phrase, each located at it (`phase_rank_bind._menu_choices`). Vetting
    # sets them.
    menu_choices: tuple[QualifierMatch, ...] = ()
    # Bind assigns this before any reader; `cast(None)` avoids an invalid sentinel or Optional checks.
    bind_result: _UniversalBindOutput = field(default=cast(_UniversalBindOutput, None))
    # The binder's whole answer to the bind call (`binders.BindResult`): its reply before
    # `split_stray_bindings`, with what the engine reads beside it. `None` before bind.
    bind_decision: BindResult | None = None
    # Shape selected by the bind call.
    shape_draft: ShapeDraft = field(default=cast(ShapeDraft, None))
    # What each bind menu shape stated of data at the places the part names, in
    # `capped_shapes` order (`phase_rank_bind._phase_bind`). Empty when the part names no
    # place, so the menu stated none.
    menu_availability: list[bool | None] = field(default_factory=list)
    # The capped-shape index of the shape in play: at election, the first shape the menu
    # states as the reading the model chose (`bind.rank_elected`,
    # `phase_election._vet_election`), and the winner's once re-election installs a
    # stand-in (`phase_place_materialise._install_shape`). Every reader of which menu
    # reading is P reads it; `bind_result.chosen_shape_index` stays the model's answer.
    # -1 before election.
    elected_index: int = -1
    # The chosen shape's own constraint bindings from the universal bind call.
    bindings: list[SlotBindingDraft] = field(default_factory=list)
    # One requirement per value the bind drafted on a property the elected shape has no
    # slot for, judged as a value the request names and named by its label
    # (`phase_rank_bind._stray_requirements`). The drafts leave the bind output, bind
    # records these in the ledger, and `_resolve_query_qualifiers` records the terms of
    # those that refuse. A part whose answer leaves one a recorded term supports unapplied
    # refuses `qualifier_not_bound` naming it, and a `qualifier_not_bound` refusal of
    # settlement, which judges no answer, names it too (`requirements.settle_unanswered`).
    # One only the model read departs instead (`Requirement.refuses`).
    stray_requirements: tuple[Requirement, ...] = ()
    # Which what and how slots the elected shape's identity carries, decided once its
    # bindings settle (`standard.breakdown_decision`, `phase_election._narrow_to_breakdown`).
    # A `Narrow` decision has narrowed the elected shape's slot keys to its properties.
    breakdown: BreakdownDecision = field(default_factory=Keep)
    # The elected shape the decision read, with every slot key: the shape itself unless a
    # `Narrow` decision narrowed it. None before the decision runs.
    breakdown_shape: ShapeDraft | None = None
    # Open constraint axes the elected measure's members do not declare; added to
    # `inapplicable_slots` because they are absent from the shape's slot keys.
    measure_dropped_axes: tuple[str, ...] = ()
    # The bind-time measure election (`bound_values.elect_measure_family`). The
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
    # Properties whose admitted qualifier matches bound the slot
    # (`phase_rank_bind._force_qualifier_bindings`); `build_spec` marks their provenance.
    text_recognized_properties: frozenset[str] = frozenset()
    # Measure properties inserted by family election; `build_spec` marks their provenance.
    measure_elected_properties: frozenset[str] = frozenset()
    # Properties defaulted by `apply_declared_defaults`; `build_spec` marks their provenance.
    defaulted_properties: frozenset[str] = frozenset()
    # What each declared default that fired states, by its property (`slot_default_phrases`).
    slot_defaults: dict[str, SlotDefaultDisclosure] = field(default_factory=dict)
    # Each named narrowing term and its outcome; build_spec records these, and
    # coverage and final assembly use them for their guards.
    query_qualifiers: list[ResolvedQualifier] = field(default_factory=list)
    # Every term the request names that each final answer is judged against
    # (`Requirement`), in the order recorded.
    requirements: tuple[Requirement, ...] = ()
    # The judgement of the part's final answers (`requirements.settle_answer`,
    # `requirements.settle_flagged`, `requirements.settle_candidate_set`), `None` until one
    # is judged. The region
    # pipeline emits the departures it leaves at its exit
    # (`requirements.with_requirement_departures`).
    judgement: Judgement | None = None
    # Caller-pinned dcids are type-checked only; their text cannot replace graph labels.
    pinned_entity_dcids: frozenset[str] = frozenset()
    # The place context of the shape in `shape_draft`: its roles, the places it is probed
    # and read over, and its place warnings. `_install_shape` sets it with the shape. Until
    # the elected shape is read it holds each place set that read decides at its members
    # (`UnsettledPlace.unread`); once the elected result stands it is the context the
    # answer is read over (`PlaceContext.settled`).
    place: PlaceContext = field(default_factory=empty_place_context)
    # The elected shape's places before its place sets settle. Materialisation settles
    # them on the shape's own read (`place_context.settle_sets`).
    unsettled_place: UnsettledPlace | None = None
    # Whether a role admits a graph type, by role name and type, as role admission
    # answered for the nodes and members of the query's place sets
    # (`phase_intake._prune_shapes_by_admission`,
    # `phase_place_materialise._unsettled_places`).
    node_admissions: dict[tuple[str, str], bool] = field(default_factory=dict)
    # Where the StatVars of the last default-scope read have data; the default scope of the
    # shape in play, when it has one, is taken from it. `None` until a shape needs one.
    default_presence: DefaultPresence | None = None
    # The elected shape's assignment of the query's place sets as admitted, before the
    # role-class election narrows it. An alternate puts a place named with no direction on
    # the axis this assignment gives it (`place_roles._undirected_place_axis`).
    elected_place_roles: PlaceAssignment | None = None
    # Each alternate's assignment of the query's place sets, by capped-shape index, and why
    # each alternate that takes none is left out. `_phase_place_bindings` assigns them once.
    alternate_place_roles: dict[int, PlaceAssignment] = field(default_factory=dict)
    unbindable_alternates: dict[int, str] = field(default_factory=dict)
    # What the status rule makes of the elected shape's menu (`status_rule.plan_status`),
    # planned once its alternates' places are bound and again once the elected shape's
    # read settles its own data (`phase_place_materialise._plan_status`).
    status_plan: StatusPlan = field(default_factory=StatusPlan)
    # The measure axes the question leaves open for the shape in play
    # (`phase_place_materialise._open_axes`), set each time the status rule plans its menu.
    # The status rule and the alternate guard both read it (`QueryPins.open_axes`). Every
    # axis is closed until the first plan.
    open_measure_axes: OpenAxes = field(default=CLOSED_AXES)
    # Why the status rule offers each alternate it plans to materialise, by capped-shape
    # index. A candidate built from one carries it (`_Candidate.basis`).
    offer_bases: dict[int, CandidateBasis] = field(default_factory=dict)
    # The elected shape read with the other direction of a place the question names without
    # one, by capped-shape index of its copy, when both directions hold data: offered beside
    # it as a `direction` candidate (`phase_place_materialise._settle_direction`).
    direction_rivals: dict[int, PlaceAssignment] = field(default_factory=dict)
    # The elected shape narrowed to another breakdown of a family with no total, by
    # capped-shape index of its copy, with its own bindings: offered beside the elected
    # breakdown as a `breakdown` candidate (`phase_place_materialise._settle_breakdown`).
    breakdown_rivals: dict[int, list[SlotBindingDraft]] = field(default_factory=dict)
    # The departures a definite answer states that no warning discloses: the direction it
    # reads a place named without one on (`RegionResult.inferred_facts`).
    inferred_facts: list[DepartureFact] = field(default_factory=list)
    # The capped-shape indices of the readings status rule 1 tries in the elected shape's
    # place when it has no observations at its places: its equivalent readings, then its
    # measurement-qualifier siblings (`phase_place_materialise._stand_in_indices`).
    reelection_siblings: list[int] = field(default_factory=list)
    # The `elected_shape_no_data` warning of a stand-in that re-election installed in the
    # elected shape's place, held in `warnings`. Its `elected` lists the stand-in's draft
    # label until a Spec is built for the stand-in, whose shape label it then lists
    # (`state_reelected_reading`).
    reelection_warning: Warning | None = None
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
    # it, whichever alternates it would have served.
    alternates_presence_failure: AlternateUnevaluatedReason | None = None
    # The alternates that passed the admission guards and were left unevaluated by that
    # failure, by capped-shape index, each with its reason. The result of the pass states
    # them (`pipeline._stating_unevaluated`).
    unevaluated_alternates: dict[int, AlternateUnevaluatedReason] = field(default_factory=dict)
    # Each default-scope taker's served and covered StatVars (`taker_stat_vars`), elected off
    # the event loop once per shape, bindings and role keys
    # (`phase_place_materialise._read_taker_stat_vars`). Each entry holds its shape, so no
    # other shape takes its id while the state lives.
    taker_stat_vars: dict[
        tuple[int, tuple[str, ...], RoleKeys], tuple[ShapeDraft, TakerStatVars]
    ] = field(default_factory=dict)
    # The siblings zero-observation re-election materialised with data and offered beside
    # the elected shape, by capped-shape index. Candidate admission takes each from here,
    # so a sibling is materialised once.
    sibling_candidates: dict[int, _Candidate] = field(default_factory=dict)
    mat_result: Materialised | NoDataDraft | MaterialisedCandidates | UnsupportedDraft | None = None
    # The elected shape's materialise result when the quantity gate found that shape does
    # not serve the leg's quantity but an admitted alternate does, so the fan-out runs. The
    # elected shape then answers nothing alone: an outcome with no other reading refuses
    # `denominator_not_available` (`phase_place_materialise.refuse_unserved_quantity`).
    quantity_refusal: (
        Materialised | NoDataDraft | MaterialisedCandidates | UnsupportedDraft | None
    ) = None
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

    def __post_init__(self) -> None:
        self.quantity_requirement = quantity_requirement(
            [self.variable, *(coded.phrase for coded in self.qualifiers or ())]
        )
        self.frame = door_frame(
            door=self.door,
            variable=self.variable,
            entities=self.entities,
            entity_roles=self.entity_roles,
            member_entities=self.member_entities,
            pins={
                name: resolution.entity.dcid
                for name, resolution in (self.pre_resolved or {}).items()
                if isinstance(resolution, PinnedResolved)
            },
            date_request=self.date_request,
            unit_text=self.unit_text,
            scope=self.scope,
            operations=self.operations,
            open_axes=self.open_axes,
        )

    @property
    def slot_default_phrases(self) -> frozenset[str]:
        """Normalised dropped phrases a named slot default answers ("flows" when the flow
        defaults to ODA); the part's `slot_defaulted` warning replaces their entry-level
        one."""
        return frozenset().union(*(default.phrases for default in self.slot_defaults.values()))

    def forget_bound(self, properties: AbstractSet[str]) -> None:
        """Drop what the engine derived from binding `properties`, once they are unbound:
        their place among the text-recognised, defaulted and measure-elected properties,
        each fired default's warnings and the phrases it answers (`slot_defaults`), and,
        when a measure-elected property goes, the measure election's partition and dropped
        axes, which describe a family the bindings no longer select."""
        if properties & self.measure_elected_properties:
            self.measure_elected_partition = None
            self.measure_dropped_axes = ()
        self.text_recognized_properties -= properties
        self.measure_elected_properties -= properties
        self.defaulted_properties -= properties
        stale = [self.slot_defaults.pop(prop) for prop in properties if prop in self.slot_defaults]
        dropped = {id(warning) for default in stale for warning in default.warnings}
        self.warnings[:] = [warning for warning in self.warnings if id(warning) not in dropped]

    @property
    def standard_place_dcids(self) -> list[str]:
        """Every place the shape in play binds, on any role, distinct and sorted by dcid
        (`place_roles.place_order`)."""
        return place_order(dcid for dcid, _role in self.place.roles.roles)

    @property
    def destination_dcid(self) -> str | None:
        """The first place the shape in play binds, by dcid, for paths that require one
        place. `None` when it binds none."""
        return next(iter(self.standard_place_dcids), None)

    @property
    def query_entity_text_by_dcid(self) -> dict[str, str]:
        """The request's text for each place its sets hold (`place_roles.place_texts`),
        so the elected Spec and an alternate label a place alike. Grounding uses it for
        display, while identity remains keyed by dcid."""
        return place_texts(self.entities, self.place_sets)

    @property
    def entity_label_text_by_dcid(self) -> dict[str, str]:
        """Return display aliases for resolved, non-pinned entities."""
        return {
            dcid: text
            for dcid, text in self.query_entity_text_by_dcid.items()
            if dcid not in self.pinned_entity_dcids
        }

    def held_codes_of(self, shape: ShapeDraft) -> Callable[[QualifierMatch], bool]:
        """`shape`'s held-value reading (`bound_values.held_codes`), built on first use."""
        entry = self.shape_held_codes.get(id(shape))
        if entry is None or entry[0] is not shape:
            entry = self.shape_held_codes[id(shape)] = (shape, held_codes(shape))
        return entry[1]

    def all_qualifier_matches(self) -> tuple[QualifierMatch, ...]:
        """Every vocabulary's qualifier matches, in vocabulary order (`merge_lists`)."""
        return merge_lists(self.query_qualifier_matches.values())

    def depart(self, warning: Warning, fact: DepartureFact) -> None:
        """Append `warning` and record `fact`, the departure it discloses, under its key."""
        self.warnings.append(warning)
        self.departure_facts[warning_key(warning)] = fact

    def state_reelected_reading(self, spec: Spec) -> None:
        """Make `elected` of the re-election warning (`reelection_warning`) the shape label
        of `spec`, the Spec built for the stand-in. The warning keeps its position in
        `warnings`. Without a re-election this does nothing."""
        warning = self.reelection_warning
        if warning is None:
            return
        stated = warning.model_copy(
            update={"params": {**(warning.params or {}), "elected": [spec.resolved.shape.label]}}
        )
        self.warnings[:] = [stated if kept is warning else kept for kept in self.warnings]
        # The candidates path can build the stand-in's Spec and hand the part back to
        # definite assembly, which states its own Spec.
        self.reelection_warning = stated

    def codes_named(self, asked: str, prop: str) -> list[GraphRef]:
        """The code rows on `prop` that the query's qualifier matches, ties and admitted
        qualifiers read `asked` as, else the rows the query term of that text names. Empty
        when neither names one (a value the request named only by its bound name)."""
        phrase = _normalize(asked)
        matches = (
            *self.measure_query_matches,
            *self.all_qualifier_matches(),
            *self.admitted_qualifiers,
            *(
                row
                for ties in self.query_qualifier_ties.values()
                for tie in ties
                for row in tie.rows
            ),
        )
        refs = {
            m.code_dcid: GraphRef(ref=m.code_dcid, label=m.label)
            for m in matches
            if m.code_system == prop and m.phrase == phrase
        }
        if not refs:
            refs = {
                ref.ref: ref
                for q in self.query_qualifiers
                if _normalize(q.text) == phrase
                for ref in q.named
            }
        return list(refs.values())

    def require_unapplied(
        self, phrase: str, *, asked: str, property_dcid: str | None, origin: RequirementOrigin
    ) -> None:
        """Record that the bindings leave the request's words `asked`, normalised `phrase`,
        unapplied: a code requirement on `property_dcid` whose codes are the rows the words
        read as there (`codes_named`), else, when no one slot clears them, a term no answer
        is judged on (`unjudged_requirement`)."""
        if property_dcid is None:
            requirement = unjudged_requirement(
                phrase, asked=asked, property_dcid=None, origin=origin
            )
        else:
            named = tuple(self.codes_named(asked, property_dcid))
            requirement = code_requirement(
                property_dcid=property_dcid,
                codes=tuple(ref.ref for ref in named),
                asked=asked,
                named=named,
                departure=unapplied_departure(phrase, asked=asked, property_dcid=property_dcid),
                origin=origin,
            )
        self.requirements = with_requirement(self.requirements, requirement)

    def require_measure(self, phrase: str, match: QualifierMatch, *, discloses: bool) -> None:
        """Record that the elected shape does not carry the measure value `match` names, by
        the request's words of normalised `phrase`, as a requirement each final answer is
        judged on by its own shape and settled bindings (`MeasureRule`). When the part
        `discloses` the phrase it departs as the code requirement `require_unapplied`
        records on the measure axis. A phrase the raw text owns (a dropped qualifier) departs
        and refuses nothing: its raw-text warning discloses it, and the part admits it when
        every final answer meets it (`requirements.met_measure_phrases`)."""
        asked, prop = match.matched_text, match.code_system
        named = tuple(self.codes_named(asked, prop))
        codes = tuple(ref.ref for ref in named)
        requirement = code_requirement(
            property_dcid=prop,
            codes=codes,
            asked=asked,
            named=named,
            departure=unapplied_departure(phrase, asked=asked, property_dcid=prop),
            origin="bind",
        )
        rule = MeasureRule(match=match, code=CodeRule(property_dcid=prop, codes=codes))
        if not discloses:
            requirement = replace(requirement, departure=None, refuses=False)
        self.requirements = with_requirement(self.requirements, replace(requirement, rule=rule))

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

    def settled_dataset(self, candidate: _Candidate | None = None) -> str | None:
        """The dataset the part's run has settled on for `candidate`, else for the elected
        shape: the one the shape names (`ShapeDraft.named_dataset`), which is the dataset
        the part's scope leaves serving it or the opt-in dataset electing it chose
        (`phase_election._name_served_dataset`). `None` when it names none, and before
        bind elects a shape. A candidate's shape names its own dataset, since vetting
        stamps every shape the part is offered.
        """
        shape = self.shape_draft if candidate is None else candidate.shape
        return shape.named_dataset if isinstance(shape, ShapeDraft) else None

    def source_code_no_data(
        self, reason: str, *, llm_usage: dict | None = None
    ) -> RegionResult | None:
        """The ``source_code_not_held`` refusal a no-data result of `reason` becomes for a part
        whose series code named its StatVars (`source_code`), or `None` for any other part and
        reason. Every no-data exit builds its result through this or `no_data`."""
        if self.source_code is None or reason != "variable_not_resolved":
            return None
        refusal = not_held(
            self.source_code.code,
            cause="unanswerable" if self.shapes_found else "not_served",
            index=get_index(),
        )
        return source_code_refusal_result(
            self.variable,
            refusal,
            timing_by_step=self.timing,
            warnings=tuple(self.warnings),
            llm_usage=llm_usage,
        )

    def no_data(
        self,
        reason: NoDataReason,
        *,
        flagged_match: tuple[Spec, ...] | None = None,
        gate: NoDataGate | None = None,
        alternate_match: Spec | None = None,
        nearest_real: tuple[NearestRealSpec, ...] | None = None,
        llm_usage: dict | None = None,
        unresolved_entity_names: tuple[str, ...] = (),
        untold_groups: tuple[tuple[str, ...], ...] = (),
        ambiguous_entities: tuple[AmbiguousEntity, ...] = (),
        resolved_entity_dcids: Mapping[str, str] | None = None,
        role_not_admitted_entities: tuple[RoleNotAdmittedEntity, ...] = (),
        cause: Cause | None = None,
        unbound_qualifiers: tuple[ResolvedQualifier, ...] = (),
        unbound_sources: tuple[ResolvedQualifier, ...] = (),
        unit_alternatives: tuple[UnitAlternative, ...] = (),
        denominator: NoDataDenominator | None = None,
        role_assignments: tuple[Mapping[str, RoleAxis], ...] = (),
        candidate: _Candidate | None = None,
    ) -> RegionResult:
        """Build a no-data result, preserving `cause`. A `no_observations` answer over an
        elected result holding data outside the years read carries its `year_unavailable`
        fact (`window_gap_fact`), whatever else the answer names. The result carries the
        dataset the run settled on (`settled_dataset`), with or without a flagged match.

        Shape-derived fields (the elected StatVars, the open role axis and the bound role
        dcids) describe `candidate` when the candidates door answers with its refusal, else
        the elected shape.

        A part whose series code named its StatVars (`source_code`) refuses
        ``source_code_not_held`` in place of ``variable_not_resolved``, since the variable is
        the code's. The cause is `not_served` when no shape derived
        from the StatVars (the catalogue admits none, or the graph confirms none) and
        `unanswerable` when one did and the part still could not be answered from it.

        The part holds `flagged_match` and `alternate_match`, so those Specs raise
        `family_total_absent`. `elected_stat_vars` names members without a
        Spec and raises none.
        """
        if (refused := self.source_code_no_data(reason, llm_usage=llm_usage)) is not None:
            return refused
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
            settled_dataset=self.settled_dataset(candidate),
            gate=gate,
            alternate_match=alternate_match,
            nearest_real=nearest_real,
            elected_stat_vars=_elected_stat_vars(
                _mat_result,
                _shape,
                allow_shape_draft_fallback=reason == "role_incompatible",
            ),
            llm_usage=llm_usage,
            unresolved_entity_names=unresolved_entity_names,
            untold_groups=untold_groups,
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
            unbound_sources=unbound_sources,
            unit_alternatives=unit_alternatives,
            denominator=denominator,
            role_assignments=role_assignments,
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
            refused_answer=(
                None if candidate is None else (candidate.shape, tuple(candidate.bindings))
            ),
        )
