"""The provider reading of a part that binds a published total with a bilateral source,
elected before Search reads the part's table.

DAC2a's total ODA is read at its bilateral destinations alone where the part names none or
only bilateral ones (`scope.provider.with_bilateral_default`), and that reading reads other
rows of the table than the total does. `elect_reading` makes the part's provider choice
from what the part holds before its read: its shape and bindings, the place the read is
tried at, the scope entries and records its binders hold, the coverage snapshot, and the
years the read reads (`extract.read_window`). The read can then read the composite's table
within its population (`ReadingDecision.table`), and the answer's scope applies the same
choice to the identity minted after the read (`scope_decision.build_part_scope`). What the
composite's table read binds beyond the part's own places, the population's total in place
of a destination the part leaves out, is bound inside the read alone (`ReadBinding`), so the
part's place, bindings and minted identity stay as the request named them.

The window is the read's own. A trailing request settles on a narrower range after the read
(`trailing.settle_trailing`), always inside the read window (`trailing._assert_inside`), and
the bilateral option is judged year by year (`provider._sibling_option`), so a composite
elected over the read window covers every range the answer can settle on."""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from code_index.composites import composite_ref
from code_index.members import (
    SourceNotInFamily,
    canonical_identity,
    identity_family_query,
    index_family,
)
from code_index.query import CodeIndex
from code_index.scope_conventions import BILATERAL_SOURCES
from code_index.shared import get_index
from dfo_platform.roles import role_key_axis
from origin_catalogue import bound_catalogue, current_catalogue
from search.engine.degraded_reads import note_degraded_read
from search.engine.errors import EngineInputError
from search.engine.extract import read_window
from search.engine.identity_edit import TableRead, composite_table_identity, open_within
from search.engine.region.codes import ROLE_SCOPE_DEFAULTED
from search.engine.region.grounding import requested_window_from_date_request
from search.engine.region.scope_decision import (
    READ_WINDOW_NOT_COVERED,
    ProviderChoice,
    ScopeDecision,
    answer_scope_facts,
    build_part_scope,
    choose_provider,
    guarded_scope_decision,
    note_undecided,
    undecided_choice,
)
from search.engine.region.state import MintedComposite
from search.engine.retrieval.bound_values import bound_source
from search.engine.retrieval.retrieve import NoDataDraft
from search.engine.shape import entities_by_role, identity_shape_for
from search.engine.spec_id import _requested_window_member, identity_spec_id
from search.models import (
    BaseIdentity,
    EntityRoleKey,
    GraphRef,
    IdentityWindow,
    Operation,
    Spec,
    Warning,
)
from search.scope.provider import Withheld, counted_source
from search.scope.state import ScopeState
from search.scope.tables import PartFacts, ScopeFacts
from series_coverage.shared import CoverageSnapshot

if TYPE_CHECKING:
    from materialise.roles import RoleAxis
    from origin_catalogue import Catalogue
    from search.engine.bind import SlotBindingDraft
    from search.engine.extract import DateRequest
    from search.engine.region.place_context import PlaceContext
    from search.engine.retrieval.standard import FacetBatch
    from search.engine.shape import ShapeDraft
    from search.engine.spec_build import AnswerIdentity

logger = logging.getLogger(__name__)

_ELECTION = "reading election"
"""The subject an election that raises is logged under (`scope_decision.note_undecided`),
and the read an election's fault or refusal is noted as (`note_degraded_read`)."""

_OPTION_NOT_COVERED = Withheld(why="option_not_covered")
"""The reason of a reading the election withheld because the bilateral destinations do not
cover every year the read reads (`provider.bilateral_default_withheld`)."""

_BILATERAL_CODE_SYSTEMS = frozenset(row.code_system for row in BILATERAL_SOURCES)

_REFUSALS: ContextVar[set[tuple[str | None, str | None, str | None]] | None] = ContextVar(
    "search_refused_composites", default=None
)
"""The refusals of composites' table reads met in the bound resolution
(`noting_refusals_once`), each the composite's ref with the refusal's code and pointer, so
a refusal is logged and noted once however many places, windows and shapes elect the
composite. The set is shared by the threads the resolution starts, so `_refusals_lock`
guards it."""

_refusals_lock = threading.Lock()


@contextmanager
def noting_refusals_once() -> Iterator[None]:
    """Bind a fresh record of refused composites for one region's resolution
    (`pipeline.resolve_variable`), so each composite whose table read refuses is logged at
    ERROR and noted as a degraded read once there however many of its reads elect it: the
    answer's, each alternate's, each stand-in's and each nearest-real suggestion's. A region
    binds its own, so each region's record of degraded reads, which decides whether a cache
    stores it, holds the note. Outside a binding every refusal is logged and noted."""
    token = _REFUSALS.set(set())
    try:
        yield
    finally:
        _REFUSALS.reset(token)


def _first_refusal(composite: str | None, refusal: EngineInputError) -> bool:
    """Whether `refusal` of composite `composite`'s table read is the first such the bound
    resolution meets, recording it. True outside a binding (`noting_refusals_once`)."""
    refused = _REFUSALS.get()
    if refused is None:
        return True
    with _refusals_lock:
        key = (composite, refusal.code, refusal.pointer)
        if key in refused:
            return False
        refused.add(key)
        return True


@dataclasses.dataclass(frozen=True)
class ReadingDecision:
    """The provider reading `elect_reading` elects for a part before its read.

    `choice` is the provider choice (`scope_decision.ProviderChoice`): the composite the
    bilateral default reads or None where the part reads the table, why the default could
    not apply, the provider record and the selection its scope reads. `table` is what the
    read reads for the composite the part's scope compiles to
    (`identity_edit.composite_table_identity`): its table identity, with the populations its
    roles bind and the sides it reads. None where the part reads no composite."""

    choice: ProviderChoice
    table: TableRead | None
    # The entities of `table.identity`, by role key, on each flow axis the part neither
    # names nor opens: the population's total a component binds in place of a destination
    # the part leaves out (`RolePopulation.total`). The read binds them as the composite's
    # table read does (`ReadBinding`), and the part's place and minted identity leave them out.
    totals: Mapping[str, tuple[str, ...]] = dataclasses.field(default_factory=dict)
    # Whether the composite the part's scope compiles to refuses its table read
    # (`identity_edit.composite_table_identity`), so the part reads nothing and answers
    # no_data, `facet_not_available`: a fresh search does not refuse its scope, and the
    # whole table's rows would answer under the composite. Its `choice` is then an undecided
    # reading's (`scope_decision.undecided_choice`), so a flagged match built over the part,
    # which reads nothing, is the table's Spec and says it reads total ODA.
    refused: bool = False
    # Whether the election raised (`elect_reading`), so it chose nothing and the part reads
    # the table. The answer's scope then judges over the answer's own identity whether the
    # bilateral default reaches it (`decide_read_scope`, `scope_decision.undecided_choice`).
    undecided: bool = False
    # The window of the document the election read (`_pre_read_identity`): the years the
    # read reads, which a trailing request's answer narrows (`_window_explains`). None where
    # the election refused or raised.
    window: IdentityWindow | None = None


class ElectionFacts:
    """The scope facts the elections of one part's reads read (`elect_reading`): the
    request's `coverage`, the part's `operations` and the dataset the code index elects for
    each election's document. One `ScopeFacts` is held per dataset, so the elections of
    every place a read tries (`ElectedReads`), of a part's alternates and of its nearest-real
    relaxations share the family scans and coverage reads it memoises, each keyed by what it
    reads. The elections run one at a time, as a part's reads do."""

    def __init__(self, *, coverage: CoverageSnapshot, operations: Sequence[Operation]) -> None:
        self.coverage = coverage
        self._operations = tuple(operations)
        self._held: dict[str | None, ScopeFacts] = {}

    def facts(self, index: CodeIndex, dataset: str | None) -> ScopeFacts:
        """The facts of a part served from `dataset`, read in `index`; a part serving no
        dataset yet where it is None."""
        held = self._held.get(dataset)
        if held is None or held.index is not index:
            datasets = () if dataset is None else (dataset,)
            held = ScopeFacts(
                index=index,
                coverage=self.coverage,
                part=PartFacts(operations=self._operations, datasets=datasets),
            )
            self._held[dataset] = held
        return held


def elect_reading(
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    place: PlaceContext,
    *,
    date_request: DateRequest | None,
    selection: ScopeState,
    facts: ElectionFacts,
) -> ReadingDecision | None:
    """The provider reading of the part read for `shape` under `bindings` at `place`, one
    settled place the read tries, with the scope entries and records of `selection`, judged
    in `facts`' coverage over the years the read reads for `date_request`
    (`extract.read_window`). None where the document the part's pre-read state names
    (`_pre_read_identity`) counts no published total with a bilateral source, such as
    DAC2a's aid type 206, by the rule the answer's scope reads after the read
    (`provider.counted_source`).

    The provider rules read an identity, so the election reads that document. Its choice
    (`scope_decision.choose_provider`) is built into a scope over that document to name the
    composite the scope compiles to, whose table read is elected here
    (`ReadingDecision.table`), with the totals it binds on axes the part leaves out
    (`ReadingDecision.totals`).

    An election that raises anywhere, building the document included, is logged as a scope
    decision is (`scope_decision.note_undecided`) and elected as undecided
    (`ReadingDecision.undecided`): the part reads the table as total ODA, saying so where the
    bilateral default reaches the answer's identity (`decide_read_scope`). A composite that
    refuses its table read is elected as refused, with an undecided reading's
    choice (`ReadingDecision.refused`), and logged at ERROR and noted as a degraded read, so
    no cache stores the answer, once per region's resolution (`noting_refusals_once`).

    Raises `EngineInputError` (`request_invalid`) for a reversed window, as the answer's mint
    refuses it (`grounding.requested_window_from_date_request`)."""
    if not _declares_bilateral_code_system(shape):
        return None
    identity: BaseIdentity | None = None
    try:
        index = get_index()
        identity = _pre_read_identity(
            shape, bindings, place, date_request=date_request, index=index
        )
        if counted_source(identity, index) is None:
            return None
        scope_facts = facts.facts(index, _elected_dataset(identity, index))
        choice = choose_provider(identity, _snapshot(selection), scope_facts)
        decision = build_part_scope(identity, choice, spec_id=_ELECTION, facts=scope_facts)
        window = identity.requested_window
        if not decision.composite:
            return ReadingDecision(choice=choice, table=None, window=window)
        assert decision.scope is not None
        composite = decision.scope.compiled.identity
        try:
            table = composite_table_identity(
                composite, pointer="/input/identity", coverage=facts.coverage
            )
        except EngineInputError as exc:
            composite_id = identity_spec_id(composite)
            if _first_refusal(_composite_of(decision), exc):
                logger.error(
                    "reading election of %s: the composite refuses its table read (%s): %s",
                    composite_id,
                    exc.code,
                    exc,
                )
                note_degraded_read(_ELECTION)
            return ReadingDecision(
                choice=undecided_choice(
                    identity, selection=selection, facts=scope_facts, subject=_ELECTION
                ),
                table=None,
                refused=True,
            )
        return ReadingDecision(
            choice=choice, table=table, totals=_totals(identity, table), window=window
        )
    except Exception as exc:
        if identity is None and isinstance(exc, EngineInputError) and exc.code == "request_invalid":
            # A reversed window, refused building the document (`_read_window`).
            raise
        note_undecided(_ELECTION, exc)
        unchosen = ProviderChoice(
            states_scope=False, bilateral=None, withheld=None, selection=selection, provider=None
        )
        return ReadingDecision(choice=unchosen, table=None, undecided=True)


def _totals(identity: BaseIdentity, table: TableRead) -> dict[str, tuple[str, ...]]:
    """The entities `table`, the composite's table read for the part of `identity`, binds on
    a flow axis `identity` neither names nor opens, by role key, left as bound once each
    population role is open within its members (`identity_edit.open_within`): the
    population's total a component binds in place of a destination the part leaves out."""
    asked = {role_key_axis(key) for key in identity.entities} | set(identity.open_roles or ())
    opened, _within = open_within(table.identity, table.populations)
    return {
        key: tuple(dcids)
        for key, dcids in opened.entities.items()
        if (axis := role_key_axis(key)) is not None and axis not in asked
    }


def elect_at_read(
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    place: PlaceContext,
    *,
    date_request: DateRequest | None,
    selection: ScopeState,
    facts: ElectionFacts,
) -> ReadingDecision | None:
    """The provider reading a read of `shape` under `bindings` at `place` elects
    (`elect_reading`), as every read of a part elects it: the definite answer's, each
    alternate's and each stand-in's. A reversed window elects none, so the answer's mint
    refuses it with `request_invalid`, where a read that reaches the mint refuses it."""
    try:
        return elect_reading(
            shape, bindings, place, date_request=date_request, selection=selection, facts=facts
        )
    except EngineInputError as exc:
        if exc.code != "request_invalid":
            raise
        return None


type Within = dict[RoleAxis, frozenset[str]] | None
"""The members each open axis of a read is discovered within (`ReadBinding.within`), None where it
discovers no axis within members."""


@dataclasses.dataclass(frozen=True)
class ReadBinding:
    """What a read of a part binds beyond the context it reads in, as the provider reading
    it elected names it (`read_binding`): `within`, the members each open axis is discovered
    within (`graph_confirm_resolve`), and `totals`, the population's total a component binds
    on each flow axis the part leaves out (`ReadingDecision.totals`). A read of a composite
    binds both, as a resubmit of the composite reads its table; a read of the table binds
    neither. The totals enter the read's entities alone, so the part's place, bindings and
    minted identity name what the request named."""

    within: Within = None
    totals: Mapping[RoleAxis, tuple[str, ...]] = dataclasses.field(default_factory=dict)

    def places(self, axis: RoleAxis, dcids: Sequence[str]) -> list[str]:
        """`dcids`, the places a read binds on `axis`, followed by the totals it binds there."""
        return [*dcids, *(dcid for dcid in self.totals.get(axis, ()) if dcid not in dcids)]

    def entities(self, shape: ShapeDraft, place: PlaceContext) -> dict[str, list[str]]:
        """The entities by role name a read of `shape` at `place` binds: the context's own
        (`PlaceRoles.resolved_entities`) and the totals, each on its axis."""
        entities = place.roles.resolved_entities
        if not self.totals:
            return entities
        bound = {name: list(dcids) for name, dcids in entities.items()}
        for name, dcids in entities_by_role(shape=shape, by_axis=self.totals).items():
            held = bound.setdefault(name, [])
            held.extend(dcid for dcid in dcids if dcid not in held)
        return bound


def read_binding(reading: ReadingDecision | None) -> ReadBinding:
    """What a read elected as `reading` binds beyond its context (`ReadBinding`): for a
    composite, each role its table read (`ReadingDecision.table`) binds to a population open
    within the population's members (`identity_edit.open_within`), and the totals it binds
    on the axes the part leaves out (`ReadingDecision.totals`). Nothing for any other
    reading."""
    if reading is None or reading.table is None:
        return ReadBinding()
    _opened, within = open_within(reading.table.identity, reading.table.populations)
    totals: dict[RoleAxis, tuple[str, ...]] = {}
    for key, dcids in reading.totals.items():
        axis = role_key_axis(key)
        assert axis is not None, f"{key!r} names no flow axis"
        totals[axis] = (*totals.get(axis, ()), *dcids)
    return ReadBinding(within=within, totals=totals)


def read_totals(reading: ReadingDecision | None) -> frozenset[tuple[str, str]]:
    """Each total the read elected as `reading` binds on an axis the part leaves out
    (`ReadingDecision.totals`), as a dcid and its role key. The reading's Spec lists them
    among its entities (`roles_read`), and its identity leaves them out."""
    if reading is None or reading.table is None:
        return frozenset()
    return frozenset((dcid, key) for key, dcids in reading.totals.items() for dcid in dcids)


def named_entity_data(
    entity_data: Sequence[tuple[EntityRoleKey, GraphRef, GraphRef | None]],
    reading: ReadingDecision | None,
) -> list[tuple[EntityRoleKey, GraphRef, GraphRef | None]]:
    """`entity_data`, a read's grounded entities by role, without the totals the read
    elected as `reading` binds (`read_totals`): the entities its identity is built from
    (`grounding.build_identity_entities`), which name what the request named."""
    totals = read_totals(reading)
    return [entity for entity in entity_data if (entity[1].ref, entity[0]) not in totals]


def roles_read(
    roles: Sequence[tuple[str, EntityRoleKey]], reading: ReadingDecision | None
) -> list[tuple[str, EntityRoleKey]]:
    """`roles`, a context's wire roles, each kept in order, with each total the read elected
    as `reading` binds (`read_totals`) that `roles` does not hold: the entities the
    reading's Spec lists. A destination total goes before the first origin, as a resubmit of
    the composite lists the places of its table identity destinations first
    (`promote_places._posted_place_roles`), and an origin total goes last. A total binds on
    an axis the part leaves out, so this lists destinations first wherever `roles` holds
    flow roles alone. Each total binds on a flow axis (`ReadingDecision.totals`)."""
    listed = list(roles)
    for dcid, key in sorted(read_totals(reading)):
        axis = role_key_axis(key)
        assert axis is not None, f"total {dcid!r} binds {key!r}, no flow axis"
        if (dcid, axis) in listed:
            continue
        at = len(listed)
        if axis == "destination":
            at = next((i for i, (_dcid, role) in enumerate(listed) if role == "origin"), at)
        listed.insert(at, (dcid, axis))
    return listed


type WithinRead[T] = Callable[[PlaceContext, FacetBatch | None, ReadBinding], T]
"""A read of a part at the context it reads in, with its facet batch and what it binds
beyond that context (`ReadBinding`)."""


class ElectedReads[T]:
    """The read of `shape` under `bindings` at each place a place-set settlement tries
    (`place_context.settle_sets`), the definite answer's, each alternate's and each
    nearest-real suggestion's. Each read elects its provider reading first, at the place it
    tries (`elect_at_read`), with the scope facts the part's elections share (`facts`), and
    reads within the composite's population that reading names (`read_binding`) through `read`,
    so a part that settles on a composite is read once, as a resubmit of the composite reads
    it. A composite that refuses its table read reads nothing and answers no_data,
    `facet_not_available`.

    Each place tried is held by identity with its election, so its identity is not reused
    while the settlement runs, and the place the sets settle on gives its election
    (`settled`). `bindings` is read at each call, so a caller that edits the list between
    rounds reads the edited bindings."""

    def __init__(
        self,
        shape: ShapeDraft,
        bindings: Sequence[SlotBindingDraft],
        *,
        date_request: DateRequest | None,
        selection: ScopeState,
        facts: ElectionFacts,
        read: WithinRead[T],
    ) -> None:
        self._shape = shape
        self._bindings = bindings
        self._date_request = date_request
        self._selection = selection
        self._facts = facts
        self._read = read
        self._tried: dict[int, tuple[PlaceContext, ReadingDecision | None]] = {}

    def __call__(self, place: PlaceContext, facet_batch: FacetBatch | None) -> T | NoDataDraft:
        reading = elect_at_read(
            self._shape,
            self._bindings,
            place,
            date_request=self._date_request,
            selection=self._selection,
            facts=self._facts,
        )
        self._tried[id(place)] = (place, reading)
        if reading is not None and reading.refused:
            return NoDataDraft(reason="facet_not_available")
        return self._read(place, facet_batch, read_binding(reading))

    def settled(self, place: PlaceContext) -> ReadingDecision | None:
        """The reading the read at `place`, a place this settlement tried, elected."""
        return self._tried[id(place)][1]


@dataclasses.dataclass(frozen=True)
class ReadScope:
    """The identity `build_spec` mints a reading's Spec from (`answer`), the scope decided
    over it (`decision`), and the composite the Spec is minted as where that scope compiles
    one (`minted`), as `decide_read_scope` decides them. `decision` is None where the
    reading was read within a composite's population and no scope minting that composite
    could be decided, so its rows have no label to stand under, and every producer answers
    no_data `facet_not_available` in its place (`phase_answer.failed_closed`): the definite
    answer (`phase_answer._gather_definite_coverage`), each candidate
    (`candidate_grounding.ground_candidates`) and a flagged match
    (`phase_shared._elected_flagged_match`)."""

    answer: AnswerIdentity
    decision: ScopeDecision | None
    minted: MintedComposite | None

    def warnings(self, spec: Spec, stated: Sequence[Warning]) -> tuple[Warning, ...]:
        """The warnings of the reading `spec`, the Spec minted from this scope, shown beside
        a part's own answer, each stamped with `spec`'s spec_id. Where `spec` is minted as a
        composite, they hold each `role_scope_defaulted` warning of `stated`, the warnings the
        read it was elected at raised, counting the composite's parties
        (`CompositeMint.role_scope_warnings`). They also hold the decision's
        `bilateral_reading_unavailable` (`ScopeDecision.warning`)."""
        scoped: list[Warning] = []
        if self.minted is not None:
            defaulted = [warning for warning in stated if warning.code == ROLE_SCOPE_DEFAULTED]
            scoped.extend(self.minted.mint.role_scope_warnings(defaulted, spec))
        if self.decision is not None and self.decision.warning is not None:
            scoped.append(self.decision.warning)
        return tuple(warning.model_copy(update={"spec_id": spec.spec_id}) for warning in scoped)


def decide_read_scope(
    answer: AnswerIdentity,
    reading: ReadingDecision | None,
    *,
    selection: ScopeState,
    operations: Sequence[Operation],
    coverage: CoverageSnapshot,
    catalogue: Catalogue | None,
) -> ReadScope:
    """The scope of the reading `answer` names, whose read elected `reading`
    (`elect_at_read`), and the composite its Spec is minted as: the definite answer's
    (`phase_answer._decide_answer_scope`), each candidate's
    (`candidate_grounding._candidate_answer`), each nearest-real suggestion's
    (`grounding._relaxed_binding_suggestion`) and a flagged match's
    (`phase_shared._elected_flagged_match`). `selection` holds the part's scope entries and
    records, `operations` its operations, and the composite is minted under `catalogue`,
    else the one in force.

    The scope applies the provider choice `reading` elected (`scope_decision.build_part_scope`)
    and chooses nothing, so the reading that labels the Spec is the one that read its rows.
    A reading that elected none applies the choice its own identity makes
    (`_unelected_choice`). A reading whose election raised (`ReadingDecision.undecided`)
    reads the table as total ODA, saying so where the bilateral default reaches `answer`'s
    identity (`scope_decision.undecided_choice`). The scope is built over the reading's own identity, which names
    what the request named, and checked against the scope decided over it (`_checked`). A
    scope that compiles a composite mints the Spec as it (`MintedComposite`), with the
    composite's components listed on its compiled scope and the identity its rows were read
    for (`_read_identity`). A decision that raises leaves the reading as a single table that
    states no scope (`scope_decision.guarded_scope_decision`).

    A read made within a composite's population (`ReadingDecision.table`) holds that
    population's rows alone, so it fails closed: where its scope cannot be decided, the mint
    raises, or the scope compiles no composite, no table label stands over those rows. It is
    logged at ERROR, noted as a degraded read so no cache stores it, and the decision is
    None (`_unminted`)."""
    # Deferred: resubmit imports the answer phase, which imports this module.
    from search.engine.region.resubmit import composite_mint

    identity = answer.identity
    minted: list[MintedComposite] = []

    def decide() -> ScopeDecision:
        facts = answer_scope_facts(
            entity_groups=answer.entity_groups,
            facets=answer.facets or {},
            sources=answer.resolved_sources,
            operations=operations,
            coverage=coverage,
        )
        if reading is None:
            choice = _unelected_choice(answer.spec_id, identity, selection=selection, facts=facts)
        elif reading.undecided:
            choice = undecided_choice(
                identity, selection=selection, facts=facts, subject=answer.spec_id
            )
        else:
            choice = reading.choice
        decision = build_part_scope(identity, choice, spec_id=answer.spec_id, facts=facts)
        if reading is not None:
            decision = _checked(decision, identity, reading, selection=selection, facts=facts)
        built = decision.scope
        if not decision.composite or built is None:
            return decision
        with bound_catalogue(catalogue or current_catalogue()):
            mint = composite_mint(built.compiled.identity, coverage=coverage)
        compiled = built.compiled.model_copy(update={"components": mint.components()})
        minted.append(
            MintedComposite(
                mint=mint,
                read=_read_identity(identity, reading),
                populations=reading.table.populations
                if reading is not None and reading.table is not None
                else {},
            )
        )
        return dataclasses.replace(
            decision,
            spec_id=mint.spec_id,
            scope=built.model_copy(update={"compiled": compiled}),
        )

    if reading is not None and reading.table is not None:
        try:
            decision = decide()
        except Exception as exc:
            return _unminted(answer, exc)
        if not decision.composite:
            return _unminted(answer, None)
        return ReadScope(answer=answer, decision=decision, minted=minted[-1])
    decision = guarded_scope_decision(
        answer.spec_id,
        decide,
        identity=identity,
        selection=selection,
        operations=operations,
        coverage=coverage,
    )
    return ReadScope(
        answer=answer, decision=decision, minted=minted[-1] if decision.composite else None
    )


def _unminted(answer: AnswerIdentity, exc: Exception | None) -> ReadScope:
    """The scope of the reading `answer` names, elected to read within a composite's
    population, whose scope minting that composite raised `exc` or, with `exc` None,
    compiled no composite: no decision (`decide_read_scope`). A flagged match whose part read
    nothing reaches here too and drops the degraded-read note
    (`phase_shared._flagged_scope`), so the log states the election alone."""
    logger.error(
        "reading %s was elected within a composite's population, but %s, so it has no label",
        answer.spec_id,
        "its scope compiles no composite" if exc is None else "the composite could not be minted",
        exc_info=exc,
    )
    note_degraded_read("composite mint")
    return ReadScope(answer=answer, decision=None, minted=None)


def _read_identity(identity: BaseIdentity, reading: ReadingDecision | None) -> BaseIdentity:
    """`identity`, the identity minted from the read elected as `reading`, with the totals
    the read bound on the axes the part leaves out (`ReadingDecision.totals`): the identity
    its rows were read for, as a resubmit of the composite reads its table identity."""
    if reading is None or not reading.totals:
        return identity
    document = identity.model_dump(exclude_none=True)
    document["entities"] = {**identity.entities, **{k: list(v) for k, v in reading.totals.items()}}
    return BaseIdentity.model_validate(document)


def _unelected_choice(
    spec_id: str, identity: BaseIdentity, *, selection: ScopeState, facts: ScopeFacts
) -> ProviderChoice:
    """The provider choice of the reading `spec_id` of `identity`, whose read elected no
    reading: the one `identity` itself makes under `selection`
    (`scope_decision.choose_provider`). A choice that reads the bilateral default would label
    rows the read did not read within the composite's population, so it is a fault: it is
    logged at ERROR, noted as a degraded read so no cache stores the answer, and the reading
    reads its table as total ODA, saying so (`scope_decision.undecided_choice`)."""
    choice = choose_provider(identity, selection, facts)
    if choice.bilateral is None:
        return choice
    logger.error(
        "reading %s reads its table with no reading elected, but the bilateral default reads "
        "it as %s; it reads total ODA and is not cached",
        spec_id,
        choice.bilateral,
    )
    note_degraded_read(_ELECTION)
    return undecided_choice(identity, selection=selection, facts=facts, subject=spec_id)


def _checked(
    decision: ScopeDecision,
    identity: BaseIdentity,
    reading: ReadingDecision,
    *,
    selection: ScopeState,
    facts: ScopeFacts,
) -> ScopeDecision:
    """`decision`, built from the choice `reading` elected before the read, checked against
    the scope decided over `identity`, the reading's own identity, itself under `selection`
    (`scope_decision.decide_scope`). The check decides nothing the reading states but one
    thing: the reading was elected over the years the read reads (`extract.read_window`), so
    where the election kept the table and the range the answer settled on reads a composite,
    the read window explains it when `identity` over those years chooses as the election
    chose (`_window_explains`). The answer then reads the table it read. Where the election
    withheld the composite because it does not cover every year read (`option_not_covered`),
    its `bilateral_reading_unavailable` warning names `read_window_not_covered`; a part with
    a provider axis states the composite's years on its bilateral option.

    Any other difference, or a check that raises, is a fault in the election: it is logged at
    ERROR and noted as a degraded read, so no cache stores the answer, and the reading keeps
    the label that matches the rows it read (`_as_read`).

    A refused reading (`ReadingDecision.refused`) or an undecided one
    (`ReadingDecision.undecided`) is not checked: the election logged its refusal or fault
    and chose no reading, so its choice differs from the identity's by design."""
    if reading.refused or reading.undecided:
        return decision
    spec_id = decision.spec_id
    read_within = reading.table is not None
    try:
        choice = choose_provider(identity, selection, facts)
        # `decision` is `build_part_scope` over this identity, spec_id and facts, so where the
        # identity chooses what the election chose, the scope decided over it is `decision`
        # by construction and is not built again.
        checked = (
            decision
            if choice == reading.choice
            else build_part_scope(identity, choice, spec_id=spec_id, facts=facts)
        )
        elected, settled = _composite_of(decision), _composite_of(checked)
        if elected == settled and decision.composite == read_within:
            return decision
        explained = (
            elected is None
            and settled is not None
            and not read_within
            and _window_explains(identity, reading, selection=selection, facts=facts)
        )
    except Exception:
        logger.exception("reading check of %s raised; the answer is not cached", spec_id)
        note_degraded_read("reading check")
        return _as_read(decision, identity, read_within=read_within, selection=selection)
    if explained:
        logger.warning(
            "answer %s reads total ODA: the bilateral reading %s covers the range it settled "
            "on, not every year it read",
            spec_id,
            settled,
        )
        if reading.choice.withheld != _OPTION_NOT_COVERED:
            return decision
        unread = dataclasses.replace(reading.choice, withheld=Withheld(why=READ_WINDOW_NOT_COVERED))
        return build_part_scope(identity, unread, spec_id=spec_id, facts=facts)
    logger.error(
        "answer %s elected %s before its read%s, but its identity decides %s; it keeps the "
        "label of the rows it read and is not cached",
        spec_id,
        elected or "the table",
        " within the composite's population" if read_within else "",
        settled or "the table",
    )
    note_degraded_read(_ELECTION)
    return _as_read(decision, identity, read_within=read_within, selection=selection)


def _window_explains(
    identity: BaseIdentity, reading: ReadingDecision, *, selection: ScopeState, facts: ScopeFacts
) -> bool:
    """Whether the years the read reads explain why `reading` kept the table where
    `identity`, the answer's own identity over the narrower range it settled on, reads a
    composite: `identity` over the election's window (`ReadingDecision.window`) withholds
    what the election withheld and compiles no composite."""
    if reading.window is None:
        return False
    read = identity.model_copy(update={"requested_window": reading.window})
    choice = choose_provider(read, selection, facts)
    if choice.withheld != reading.choice.withheld:
        return False
    return _composite_of(build_part_scope(read, choice, spec_id=_ELECTION, facts=facts)) is None


def _as_read(
    decision: ScopeDecision, identity: BaseIdentity, *, read_within: bool, selection: ScopeState
) -> ScopeDecision:
    """`decision`, the scope of the reading of `identity`, where its label matches the rows
    read. A decision compiling a composite over rows read without the composite's population
    (`read_within` false) would state the whole table's rows, total ODA, as the composite's,
    so the reading reads as the table it read, saying it reads total ODA where the bilateral
    default reaches it (`scope_decision.undecided_choice`). A read within a population that
    compiles no composite has no label, which `decide_read_scope` refuses (`_unminted`)."""
    if not decision.composite or read_within:
        return decision
    spec_id, facts = decision.spec_id, decision.facts
    choice = undecided_choice(identity, selection=selection, facts=facts, subject=spec_id)
    return build_part_scope(identity, choice, spec_id=spec_id, facts=facts)


def _composite_of(decision: ScopeDecision) -> str | None:
    """The composite `decision`'s scope compiles to, None where it compiles none."""
    if not decision.composite or decision.scope is None:
        return None
    return composite_ref(
        decision.scope.compiled.identity.model_dump(mode="json", include={"facets"})
    )


def _snapshot(selection: ScopeState) -> ScopeState:
    """`selection` as it stands, so a record a binder writes after the election leaves the
    choice's selection as it was elected."""
    return dataclasses.replace(selection, records=dict(selection.records))


def _declares_bilateral_code_system(shape: ShapeDraft) -> bool:
    """Whether `shape` declares a slot on a bilateral source's code system
    (`BILATERAL_SOURCES`), as every part whose identity can count such a source's total
    does, bound in its draft or by its dataset's invariants once canonical."""
    return any(slot.property_dcid in _BILATERAL_CODE_SYSTEMS for slot in shape.slot_keys)


def _pre_read_identity(
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    place: PlaceContext,
    *,
    date_request: DateRequest | None,
    index: CodeIndex,
) -> BaseIdentity:
    """The identity the read will mint for a part, as far as its pre-read state names it:
    the shape's five-tuple, a slot for each what and how slot key of `shape` bound as its
    draft in `bindings` binds a value or a set of two or more and unbound otherwise, as the
    answer's identity states its slots (`spec_build.build_slot`), the entities `place` binds
    by role key and the axes it leaves open, the read window (`_read_window`), and a
    provenance pin to the dataset the bindings or the shape name. It is in canonical form
    (`code_index.members.canonical_identity`), so each slot an invariant of that dataset
    fixes binds the invariant's value, as in the minted identity.

    It is no answer identity: the mint after the read settles the window, which does not
    move the provider choice."""
    drafts = {binding.property_dcid: binding for binding in bindings if binding.property_dcid}
    slots = []
    for key in shape.slot_keys:
        if key.axis not in ("what", "how") or not key.property_dcid:
            continue
        binding = drafts.get(key.property_dcid)
        values = [] if binding is None else binding.value_dcids
        member: dict[str, object] = {"state": "unbound"}
        if binding is not None and binding.kind == "value" and values:
            member = {"state": "value", "ref": values[0]}
        elif binding is not None and binding.kind == "set" and len(values) >= 2:
            member = {"state": "set", "refs": sorted(values)}
        slots.append({"property": key.property_dcid, "binding": member})
    entities: dict[str, list[str]] = {}
    for dcid, key in place.roles.roles:
        held = entities.setdefault(key, [])
        if dcid not in held:
            held.append(dcid)
    document: dict[str, object] = {
        "identity_version": "spec1",
        "shape": identity_shape_for(
            pop_type_dcid=shape.pop_type_dcid,
            meas_prop_dcid=shape.meas_prop_dcid,
            stat_type_dcid=shape.stat_type_dcid,
            meas_qual_dcid=shape.meas_qual_dcid,
            meas_denom_dcid=shape.meas_denom_dcid,
            meas_denom_unresolved=shape.meas_denom_unresolved,
        ),
        "slots": sorted(slots, key=lambda slot: str(slot["property"])),
        "requested_window": _read_window(date_request),
    }
    if entities:
        document["entities"] = entities
    if place.roles.open_axes:
        document["open_roles"] = sorted(place.roles.open_axes)
    if dataset := bound_source(bindings) or shape.named_dataset:
        document["facets"] = {"provenance": {"state": "value", "ref": dataset}}
    return BaseIdentity.model_validate(canonical_identity(index, document))


def _read_window(date_request: DateRequest | None) -> dict:
    """The identity's `requested_window` member for the years the part's read reads
    (`extract.read_window`): for a trailing span, every range it can settle on; otherwise
    the window the answer's identity requests (`grounding.requested_window_from_date_request`,
    which refuses a reversed window), the latest year, or every year."""
    if date_request is not None and date_request.trailing is not None:
        return _requested_window_member(read_window(date_request))
    return _requested_window_member(requested_window_from_date_request(date_request))


def _elected_dataset(identity: BaseIdentity, index: CodeIndex) -> str | None:
    """The dataset the index elects for `identity`'s family, None where it elects none, as
    the answer's provenance facet names it after the read."""
    try:
        family = index_family(
            index, identity_family_query(index, identity.model_dump(mode="json", exclude_none=True))
        )
    except SourceNotInFamily:
        return None
    return family.provenance if family is not None else None
