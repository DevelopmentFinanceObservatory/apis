"""The provider reading of a part that binds a published total with a bilateral source,
elected before Search reads the part's table.

DAC2a's total ODA is read at its bilateral destinations alone where the part names none or
only bilateral ones (`scope.provider.with_bilateral_default`), and that reading reads other
rows of the table than the total does. `elect_reading` makes the part's provider choice
from what the part holds before its read: its shape and bindings, the place the read is
tried at, the scope entries and records its binders hold, the coverage snapshot, and the
years the read reads (`extract.read_window`). The read can then read the composite's table
within its population (`ReadingDecision.table`), and the answer's scope applies the same
choice to the identity minted after the read (`scope_decision.build_part_scope`).

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
from code_index.members import SourceNotInFamily, identity_family_query, index_family
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
from search.engine.spec_id import identity_spec_id
from search.models import BaseIdentity, Operation, Spec, TimeWindow, Warning
from search.scope.provider import Withheld
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

_BILATERAL_CODES = frozenset((row.code_system, row.code) for row in BILATERAL_SOURCES)

_REFUSALS: ContextVar[set[str] | None] = ContextVar("search_refused_composites", default=None)
"""The spec_ids of the composites whose table read refused in the bound resolution
(`noting_refusals_once`), each logged and noted once. The set is shared by the threads the
resolution starts, so `_refusals_lock` guards it."""

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


def _first_refusal(composite: str) -> bool:
    """Whether `composite`'s refusal is the first the bound resolution meets, recording it.
    True outside a binding (`noting_refusals_once`)."""
    refused = _REFUSALS.get()
    if refused is None:
        return True
    with _refusals_lock:
        if composite in refused:
            return False
        refused.add(composite)
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
    # The `entities` keys of `table.identity` on a flow axis the part neither names nor
    # opens: the population's total a component binds in place of a destination the part
    # leaves out (`RolePopulation.total`), which the read names as the composite's table
    # read does and the composite's own identity leaves out.
    left_out: frozenset[str] = frozenset()
    # Whether the composite the part's scope compiles to refuses its table read
    # (`identity_edit.composite_table_identity`), so the part reads nothing and answers
    # no_data, `facet_not_available`: a fresh search does not refuse its scope, and the
    # whole table's rows would answer under the composite. Its `choice` is then an undecided
    # reading's (`scope_decision.undecided_choice`), so a flagged match built over the part,
    # which reads nothing, is the table's Spec and says it reads total ODA.
    refused: bool = False


def elect_reading(
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    place: PlaceContext,
    *,
    date_request: DateRequest | None,
    selection: ScopeState,
    coverage: CoverageSnapshot,
    operations: Sequence[Operation],
) -> ReadingDecision | None:
    """The provider reading of the part read for `shape` under `bindings` at `place`, one
    settled place the read tries, with the scope entries and records of `selection`, judged
    in `coverage` over the years the read reads for `date_request` (`extract.read_window`).
    None where no binding counts a published total with a bilateral source
    (`BILATERAL_SOURCES`), such as DAC2a's aid type 206.

    The provider rules read an identity, so the election reads the document the part's
    pre-read state names (`_pre_read_identity`). Its choice
    (`scope_decision.choose_provider`) is built into a scope over that document to name the
    composite the scope compiles to, whose table read is elected here
    (`ReadingDecision.table`).

    An election that raises anywhere, building the document included, is logged as a scope
    decision is (`scope_decision.note_undecided`), and the part reads the table as total
    ODA, saying so where the bilateral default reaches it (`_undecided_election`). A
    composite that refuses its table read is elected as refused, with an undecided reading's
    choice (`ReadingDecision.refused`), and logged at ERROR and noted as a degraded read, so
    no cache stores the answer, once per region's resolution (`noting_refusals_once`).

    Raises `EngineInputError` (`request_invalid`) for a reversed window, as the answer's mint
    refuses it (`grounding.requested_window_from_date_request`)."""
    if not binds_bilateral_total(bindings):
        return None
    identity: BaseIdentity | None = None
    facts: ScopeFacts | None = None
    try:
        index = get_index()
        identity = _pre_read_identity(shape, bindings, place, date_request=date_request)
        # The facts of a part serving no dataset yet, which an election that raises past here
        # reads the bilateral default's reach in (`_undecided_election`).
        facts = ScopeFacts(
            index=index, coverage=coverage, part=PartFacts(operations=tuple(operations))
        )
        if (dataset := _elected_dataset(identity, index)) is not None:
            facts = dataclasses.replace(
                facts, part=dataclasses.replace(facts.part, datasets=(dataset,))
            )
        choice = choose_provider(identity, _snapshot(selection), facts)
        decision = build_part_scope(identity, choice, spec_id=_ELECTION, facts=facts)
        if not decision.composite:
            return ReadingDecision(choice=choice, table=None)
        assert decision.scope is not None
        composite = decision.scope.compiled.identity
        try:
            table = composite_table_identity(
                composite, pointer="/input/identity", coverage=coverage
            )
        except EngineInputError as exc:
            composite_id = identity_spec_id(composite)
            if _first_refusal(composite_id):
                logger.error(
                    "reading election of %s: the composite refuses its table read (%s): %s",
                    composite_id,
                    exc.code,
                    exc,
                )
                note_degraded_read(_ELECTION)
            return ReadingDecision(
                choice=undecided_choice(
                    identity, selection=selection, facts=facts, subject=_ELECTION
                ),
                table=None,
                refused=True,
            )
        asked = {role_key_axis(key) for key in identity.entities} | set(identity.open_roles or ())
        left_out = frozenset(
            key for key in table.identity.entities if role_key_axis(key) not in asked
        )
        return ReadingDecision(choice=choice, table=table, left_out=left_out)
    except Exception as exc:
        if identity is None and isinstance(exc, EngineInputError) and exc.code == "request_invalid":
            # A reversed window, refused building the document (`_read_window`).
            raise
        note_undecided(_ELECTION, exc)
        return ReadingDecision(
            choice=_undecided_election(identity, selection=selection, facts=facts), table=None
        )


def _undecided_election(
    identity: BaseIdentity | None, *, selection: ScopeState, facts: ScopeFacts | None
) -> ProviderChoice:
    """The provider choice of a part whose election raised (`elect_reading`): the table as
    total ODA, withheld with `why` `scope_undecided` where the bilateral default reaches
    `identity` (`scope_decision.undecided_choice`). Where the election raised before its
    document was built, the reach cannot be read, so it withholds nothing, as a reach check
    that raises withholds nothing."""
    if identity is None or facts is None:
        return ProviderChoice(
            states_scope=False, bilateral=None, withheld=None, selection=selection, provider=None
        )
    return undecided_choice(identity, selection=selection, facts=facts, subject=_ELECTION)


def elect_at_read(
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    place: PlaceContext,
    *,
    date_request: DateRequest | None,
    selection: ScopeState,
    coverage: CoverageSnapshot,
    operations: Sequence[Operation],
) -> ReadingDecision | None:
    """The provider reading a read of `shape` under `bindings` at `place` elects
    (`elect_reading`), as every read of a part elects it: the definite answer's, each
    alternate's and each stand-in's. A reversed window elects none, so the answer's mint
    refuses it with `request_invalid`, where a read that reaches the mint refuses it."""
    try:
        return elect_reading(
            shape,
            bindings,
            place,
            date_request=date_request,
            selection=selection,
            coverage=coverage,
            operations=operations,
        )
    except EngineInputError as exc:
        if exc.code != "request_invalid":
            raise
        return None


def read_within(
    shape: ShapeDraft, place: PlaceContext, reading: ReadingDecision | None
) -> tuple[PlaceContext, Within]:
    """The context a read of `shape` elected as `reading` at `place` reads in, and the
    members each open axis is discovered within (`graph_confirm_resolve`): for a composite,
    its table read (`ReadingDecision.table`) with each role bound to a population open within
    its members (`identity_edit.open_within`), and the population's total a component binds
    in place of a destination the part leaves out named on its axis
    (`ReadingDecision.left_out`), as a resubmit of the composite reads it. `place` itself,
    discovering no axis within members, for any other reading."""
    if reading is None or reading.table is None:
        return place, None
    opened, within = open_within(reading.table.identity, reading.table.populations)
    if not reading.left_out:
        return place, within
    return _naming(shape, place, {key: opened.entities[key] for key in reading.left_out}), within


type Within = dict[RoleAxis, frozenset[str]] | None
"""The members each open axis of a read is discovered within (`read_within`), None where it
discovers no axis within members."""

type WithinRead[T] = Callable[[PlaceContext, FacetBatch | None, Within], T]
"""A read of a part at the context it reads in, with its facet batch and the members each
open axis is discovered within."""


class ElectedReads[T]:
    """The read of `shape` under `bindings` at each place a place-set settlement tries
    (`place_context.settle_sets`), the definite answer's, each alternate's and each
    nearest-real suggestion's. Each read elects its provider reading first, at the place it
    tries (`elect_at_read`), and reads within the composite's population that reading names
    (`read_within`) through `read`, so a part that settles on a composite is read once, as a
    resubmit of the composite reads it. A composite that refuses its table read reads nothing
    and answers no_data, `facet_not_available`.

    Each place tried is held by identity with the context it was read in and its election,
    so its identity is not reused while the settlement runs, and the place the sets settle on
    gives both (`settled`). `bindings` is read at each call, so a caller that edits the list
    between rounds reads the edited bindings."""

    def __init__(
        self,
        shape: ShapeDraft,
        bindings: Sequence[SlotBindingDraft],
        *,
        date_request: DateRequest | None,
        selection: ScopeState,
        coverage: CoverageSnapshot,
        operations: Sequence[Operation],
        read: WithinRead[T],
    ) -> None:
        self._shape = shape
        self._bindings = bindings
        self._date_request = date_request
        self._selection = selection
        self._coverage = coverage
        self._operations = operations
        self._read = read
        self._tried: dict[int, tuple[PlaceContext, PlaceContext, ReadingDecision | None]] = {}

    def __call__(self, place: PlaceContext, facet_batch: FacetBatch | None) -> T | NoDataDraft:
        reading = elect_at_read(
            self._shape,
            self._bindings,
            place,
            date_request=self._date_request,
            selection=self._selection,
            coverage=self._coverage,
            operations=self._operations,
        )
        read_at, within = read_within(self._shape, place, reading)
        self._tried[id(place)] = (place, read_at, reading)
        if reading is not None and reading.refused:
            return NoDataDraft(reason="facet_not_available")
        return self._read(read_at, facet_batch, within)

    def settled(self, place: PlaceContext) -> tuple[PlaceContext, ReadingDecision | None]:
        """The context the read at `place`, a place this settlement tried, read in, and the
        reading it elected."""
        _place, read_at, reading = self._tried[id(place)]
        return read_at, reading


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
    (`_unelected_choice`). The scope is built over the identity the request names, without
    the population's total the read named in place of a destination the part leaves out
    (`request_identity`), and checked against the scope decided over it (`_checked`). A
    scope that compiles a composite mints the Spec as it (`MintedComposite`), with the
    composite's components listed on its compiled scope. A decision that raises leaves the
    reading as a single table that states no scope (`scope_decision.guarded_scope_decision`).

    A read made within a composite's population (`ReadingDecision.table`) holds that
    population's rows alone, so it fails closed: where its scope cannot be decided, the mint
    raises, or the scope compiles no composite, no table label stands over those rows. It is
    logged at ERROR, noted as a degraded read so no cache stores it, and the decision is
    None (`_unminted`)."""
    # Deferred: resubmit imports the answer phase, which imports this module.
    from search.engine.region.resubmit import composite_mint

    identity = request_identity(answer.identity, reading)
    minted: list[MintedComposite] = []

    def decide() -> ScopeDecision:
        facts = answer_scope_facts(
            entity_groups=answer.entity_groups,
            facets=answer.facets or {},
            sources=answer.resolved_sources,
            operations=operations,
            coverage=coverage,
        )
        choice = (
            _unelected_choice(answer.spec_id, identity, selection=selection, facts=facts)
            if reading is None
            else reading.choice
        )
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
                read=answer.identity,
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
    """The scope of the reading `answer` names, read within a composite's population, whose
    scope minting that composite raised `exc` or, with `exc` None, compiled no composite: no
    decision (`decide_read_scope`)."""
    logger.error(
        "reading %s was read within a composite's population, but %s; it has no label and "
        "is not cached",
        answer.spec_id,
        "its scope compiles no composite" if exc is None else "the composite could not be minted",
        exc_info=exc,
    )
    note_degraded_read("composite mint")
    return ReadScope(answer=answer, decision=None, minted=None)


def request_identity(identity: BaseIdentity, reading: ReadingDecision | None) -> BaseIdentity:
    """`identity`, the identity minted from the read, without the entities the read named in
    place of a destination the part leaves out (`ReadingDecision.left_out`): the identity the
    request names, whose composite reads that destination at the population's total."""
    if reading is None or not reading.left_out:
        return identity
    document = identity.model_dump(exclude_none=True)
    document["entities"] = {
        key: dcids for key, dcids in identity.entities.items() if key not in reading.left_out
    }
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
    thing: the reading was elected over the years the read reads (`extract.read_window`), and
    where the election withheld the composite because it does not cover every year read
    (`option_not_covered`) and the range the answer settled on reads it, its
    `bilateral_reading_unavailable` warning names `read_window_not_covered`. A part with a
    provider axis states that on its bilateral option. Any other difference, or a check that
    raises, is a fault in the election: it is logged at ERROR and noted as a degraded read,
    so no cache stores the answer, and the reading keeps `decision`, whose label matches the
    rows it read.

    A refused reading (`ReadingDecision.refused`) is not checked: the election logged its
    refusal and chose an undecided reading on purpose, so its choice differs from the
    identity's by design and is no fault of the election."""
    if reading.refused:
        return decision
    spec_id = decision.spec_id
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
    except Exception:
        logger.exception("reading check of %s raised; the answer is not cached", spec_id)
        note_degraded_read("reading check")
        return decision
    elected, settled = _composite_of(decision), _composite_of(checked)
    read_within = reading.table is not None
    if elected == settled and decision.composite == read_within:
        return decision
    if (
        elected is None
        and settled is not None
        and not read_within
        and reading.choice.withheld == _OPTION_NOT_COVERED
    ):
        logger.warning(
            "answer %s reads total ODA: the bilateral reading %s covers the range it settled "
            "on, not every year it read",
            spec_id,
            settled,
        )
        unread = dataclasses.replace(reading.choice, withheld=Withheld(why=READ_WINDOW_NOT_COVERED))
        return build_part_scope(identity, unread, spec_id=spec_id, facts=facts)
    logger.error(
        "answer %s elected %s before its read%s, but its identity decides %s; it keeps the "
        "reading it read and is not cached",
        spec_id,
        elected or "the table",
        " within the composite's population" if read_within else "",
        settled or "the table",
    )
    note_degraded_read(_ELECTION)
    return decision


def _composite_of(decision: ScopeDecision) -> str | None:
    """The composite `decision`'s scope compiles to, None where it compiles none."""
    if not decision.composite or decision.scope is None:
        return None
    return composite_ref(
        decision.scope.compiled.identity.model_dump(mode="json", include={"facets"})
    )


def _naming(
    shape: ShapeDraft, place: PlaceContext, named: Mapping[str, Sequence[str]]
) -> PlaceContext:
    """`place`, the context of role-bearing `shape`, with the places `named` maps by flow role
    key bound on that role's axis after those it binds, its destinations as its where places,
    as a posted identity's places bind (`promote_places._posted_place_roles`)."""
    roles = place.roles
    by_axis: dict[RoleAxis, list[str]] = {
        "destination": list(roles.destination_dcids),
        "origin": list(roles.origin_dcids),
    }
    for key, dcids in named.items():
        axis = role_key_axis(key)
        assert axis is not None, f"{key!r} names no flow axis"
        by_axis[axis].extend(dcid for dcid in dcids if dcid not in by_axis[axis])
    destinations, origins = by_axis["destination"], by_axis["origin"]
    bound = dataclasses.replace(
        roles,
        destination_dcids=destinations,
        origin_dcids=origins,
        resolved_entities=entities_by_role(shape=shape, by_axis=by_axis),
        roles=[(dcid, "destination") for dcid in destinations]
        + [(dcid, "origin") for dcid in origins],
        where_dcids=destinations,
        obs_destination_dcid=destinations[0] if destinations else None,
    )
    probe = None if len(destinations) <= 1 else tuple(destinations)
    return dataclasses.replace(place, roles=bound, probe_dcids=probe)


def _snapshot(selection: ScopeState) -> ScopeState:
    """`selection` as it stands, so a record a binder writes after the election leaves the
    choice's selection as it was elected."""
    return dataclasses.replace(selection, records=dict(selection.records))


def binds_bilateral_total(bindings: Sequence[SlotBindingDraft]) -> bool:
    """Whether some binding of `bindings` binds a bilateral source's total as a value
    (`BILATERAL_SOURCES`), the parts `elect_reading` elects a reading for."""
    return any(_binds_bilateral_code(binding) for binding in bindings)


def _binds_bilateral_code(binding: SlotBindingDraft) -> bool:
    """Whether `binding` binds a bilateral source's total as a value (`BILATERAL_SOURCES`)."""
    return (
        binding.kind == "value"
        and len(binding.value_dcids) == 1
        and (binding.property_dcid, binding.value_dcids[0]) in _BILATERAL_CODES
    )


def _pre_read_identity(
    shape: ShapeDraft,
    bindings: Sequence[SlotBindingDraft],
    place: PlaceContext,
    *,
    date_request: DateRequest | None,
) -> BaseIdentity:
    """The identity document the provider rules read for a part before its read: the
    shape's five-tuple, each value, set or unbound binding on the what and how axes as a
    slot, the entities `place` binds by role key and the axes it leaves open, the read window
    (`_read_window`), and a provenance pin to the dataset the bindings or the shape name.

    It is no answer identity: the mint after the read binds invariant slots and settles the
    window, and neither moves the provider choice."""
    slots = []
    for binding in bindings:
        if binding.axis not in ("what", "how") or not binding.property_dcid:
            continue
        if binding.kind == "value" and len(binding.value_dcids) == 1:
            member: dict[str, object] = {"state": "value", "ref": binding.value_dcids[0]}
        elif binding.kind == "set" and binding.value_dcids:
            member = {"state": "set", "refs": sorted(binding.value_dcids)}
        elif binding.kind == "unbound":
            member = {"state": "unbound"}
        else:
            continue
        slots.append({"property": binding.property_dcid, "binding": member})
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
    return BaseIdentity.model_validate(document)


def _read_window(date_request: DateRequest | None) -> dict[str, object]:
    """The identity's `requested_window` member for the years the part's read reads
    (`extract.read_window`): for a trailing span, every range it can settle on; otherwise
    the window the answer's identity requests (`grounding.requested_window_from_date_request`,
    which refuses a reversed window), the latest year, or every year."""
    if date_request is not None and date_request.trailing is not None:
        window = read_window(date_request)
        assert window is not None
        return _range(window)
    requested = requested_window_from_date_request(date_request)
    if requested is None:
        return {"kind": "all"}
    if requested == "latest":
        return {"kind": "latest"}
    return _range(requested)


def _range(window: TimeWindow) -> dict[str, object]:
    """The `range` member of `window`, with the bounds it sets."""
    member: dict[str, object] = {"kind": "range"}
    if window.start_year is not None:
        member["start_year"] = window.start_year
    if window.end_year is not None:
        member["end_year"] = window.end_year
    return member


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
