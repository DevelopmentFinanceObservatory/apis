"""The scope a definite answer states, decided from its identity and the part's facts
before its preview launches.

The answer phase decides it once the answer's identity is known
(`spec_build.answer_identity`), and the candidates path from its survivor's Spec, so an
answer the binders settle as a composite reads no preview of the one table it resolved.
The finished answer is checked against the decision once (`phase_answer.finished_scope`),
and the definite result carries the decision that check returns
(`RegionResult.scope_decision`), which `pipeline._with_scope_choices` states. A finished
answer no decision states is decided again from its Spec, which chooses nothing and fails
closed where it disagrees with the Spec (`decide_after_read`)."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence

from code_index.composites import CompositeNotServable, composite_ref
from code_index.shared import get_index
from search.engine.degraded_reads import LOST_READ_ERRORS, note_degraded_read
from search.engine.departures import datasets_served
from search.engine.region.codes import BILATERAL_READING_UNAVAILABLE
from search.models import (
    BaseIdentity,
    EntityGroup,
    Operation,
    PartScope,
    ProvenanceRef,
    ProviderBinding,
    ResolvedFacet,
    Spec,
    Warning,
)
from search.scope import build as scope_build
from search.scope.basis import basis_dataset, unbound_basis_record
from search.scope.edits import apply_binding
from search.scope.provider import (
    CONTRIBUTIONS,
    Withheld,
    WithheldWhy,
    bilateral_default_reaches,
    bilateral_default_reading,
    bilateral_default_withheld,
    settle_provider,
    with_summed_provider,
    without_unoffered_multilateral,
)
from search.scope.state import AxisRecord, ScopeState
from search.scope.tables import PartFacts, ScopeFacts
from series_coverage.shared import CoverageSnapshot

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ScopeDecision:
    """The scope the answer `spec_id` states, built over `facts`; None when it states none.
    `composite` when its compiled identity is a declared composite (`composite_ref`), which
    the answer's Spec is then minted as (`spec_build.build_spec(composite=)`). `warning` is the
    answer's `bilateral_reading_unavailable` warning where it reads total ODA because the
    bilateral default could not apply (`bilateral_withheld_warning`)."""

    spec_id: str
    facts: ScopeFacts
    scope: PartScope | None
    composite: bool
    warning: Warning | None = None


@dataclasses.dataclass(frozen=True)
class ProviderChoice:
    """What a fresh reading chooses of its provider scope before its scope is built
    (`choose_provider`), which `build_part_scope` applies to the identity it is given.

    `states_scope` is false where the answer states no scope at all (`states_no_scope`, with
    the bilateral default not applied). `bilateral` is the composite the bilateral default
    reads (`provider.bilateral_default_reading`), None where the reading keeps the table.
    `withheld` is why the default reaches the reading and cannot apply
    (`provider.bilateral_default_withheld`), None otherwise. `selection` is the part's
    selection with the entries the default served removed and the summed and multilateral
    words read (`provider.with_summed_provider`, `provider.without_unoffered_multilateral`),
    and `provider` the provider record settled over the served identity
    (`provider.settle_provider`), None where the axis does not apply or no scope is stated.
    The scope build writes into a copy, so `selection` keeps the entries it holds."""

    states_scope: bool
    bilateral: str | None
    withheld: Withheld | None
    selection: ScopeState
    provider: AxisRecord | None


def answer_scope_facts(
    *,
    entity_groups: Mapping[str, EntityGroup],
    facets: Mapping[str, ResolvedFacet],
    sources: Iterable[ProvenanceRef],
    operations: Sequence[Operation],
    coverage: CoverageSnapshot,
) -> ScopeFacts:
    """What the axis rules read of the part an answer stating `entity_groups`, `facets`
    and `sources` answers: the bound release's code index, `coverage`, the part's
    `operations`, the groups its roles stand for and the datasets it is served from
    (`departures.datasets_served`)."""
    return ScopeFacts(
        index=get_index(),
        coverage=coverage,
        part=PartFacts(
            operations=tuple(operations),
            entity_groups=entity_groups,
            datasets=tuple(sorted(datasets_served(facets, sources))),
        ),
    )


def scope_facts(
    spec: Spec, *, operations: Sequence[Operation], coverage: CoverageSnapshot
) -> ScopeFacts:
    """What the axis rules read of the part `spec` answers (`answer_scope_facts`)."""
    return answer_scope_facts(
        entity_groups=spec.resolved.entity_groups or {},
        facets=spec.resolved.facets,
        sources=spec.resolved.trace.resolved_sources,
        operations=operations,
        coverage=coverage,
    )


def decide_spec_scope(
    spec: Spec,
    *,
    selection: ScopeState,
    operations: Sequence[Operation],
    coverage: CoverageSnapshot,
) -> ScopeDecision:
    """The scope the finished answer `spec` states (`decide_scope`), over the facts it
    states (`scope_facts`)."""
    return decide_scope(
        spec.spec_id,
        spec.definition.identity,
        selection=selection,
        facts=scope_facts(spec, operations=operations, coverage=coverage),
    )


def guarded_scope_decision(
    spec_id: str,
    decide: Callable[[], ScopeDecision],
    *,
    identity: BaseIdentity,
    selection: ScopeState,
    operations: Sequence[Operation],
    coverage: CoverageSnapshot,
) -> ScopeDecision:
    """The decision `decide` makes for the answer `spec_id` of `identity`, or, when it
    raises, a decision that states no scope and no composite, so the answer reads as the one
    table it resolved. `decide` builds the facts it reads, so an error there is guarded too.
    Where the bilateral default reaches `identity`, that answer reads DAC2a's total ODA with
    no scope to say so, so the fallback carries the `bilateral_reading_unavailable` warning
    with `why` `scope_undecided` (`undecided_choice`).

    A composite its components cannot serve (`CompositeNotServable`) or a read the decision
    lost (`degraded_reads.LOST_READ_ERRORS`) is logged, and the answer can be stored. Any
    other error is a fault in the scope code, so it is logged with its traceback and noted
    as a degraded read (`note_degraded_read`), and the answer is served but no cache stores
    it.

    The answer phase and the candidates path decide before the carve-outs judge the answer,
    so a decision error must not fail an answer they would refuse. The fallback's facts
    are those of an empty part, which nothing reads of a decision that is not a composite."""
    try:
        return decide()
    except Exception as exc:
        note_undecided(spec_id, exc)
    facts = answer_scope_facts(
        entity_groups={}, facets={}, sources=(), operations=operations, coverage=coverage
    )
    choice = undecided_choice(identity, selection=selection, facts=facts, subject=spec_id)
    return build_part_scope(identity, choice, spec_id=spec_id, facts=facts)


def note_undecided(subject: str, exc: Exception) -> None:
    """Log `exc`, raised deciding the scope or provider choice of `subject`, whose answer
    then reads as one table that states no scope. A composite its components cannot serve
    (`CompositeNotServable`) or a read the decision lost (`degraded_reads.LOST_READ_ERRORS`)
    is a warning, and the answer can be stored. Any other error is a fault in the scope
    code, logged with its traceback and noted as a degraded read (`note_degraded_read`), so
    the answer is served but no cache stores it."""
    if isinstance(exc, (CompositeNotServable, *LOST_READ_ERRORS)):
        logger.warning(
            "scope decision for %s failed; the answer reads as one table that states no scope",
            subject,
            exc_info=exc,
        )
        return
    logger.error(
        "scope decision for %s raised; the answer reads as one table that states no scope "
        "and is not cached",
        subject,
        exc_info=exc,
    )
    note_degraded_read("scope decision")


def undecided_choice(
    identity: BaseIdentity, *, selection: ScopeState, facts: ScopeFacts, subject: str
) -> ProviderChoice:
    """The provider choice of a reading of `identity`, named `subject` in logs, whose
    choice or scope could not be decided: it states no scope and keeps the table. Where the
    bilateral default reaches `identity` (`provider.bilateral_default_reaches`) and no
    stated or carried provider entry asks for total ODA, the reading is DAC2a's total ODA
    with no scope to say so, so it is withheld with `why` `scope_undecided`. A reach check
    that raises too is logged and withholds nothing."""
    return ProviderChoice(
        states_scope=False,
        bilateral=None,
        withheld=Withheld(why="scope_undecided")
        if _undecided_reach(identity, selection, facts, subject)
        else None,
        selection=selection,
        provider=None,
    )


def _undecided_reach(
    identity: BaseIdentity, selection: ScopeState, facts: ScopeFacts, subject: str
) -> bool:
    """Whether an undecided reading of `identity` says it reads total ODA
    (`undecided_choice`)."""
    stated = selection.entries_on("provider")
    if stated and stated[0].option_id == CONTRIBUTIONS:
        return False
    try:
        return bilateral_default_reaches(identity, facts)
    except Exception:
        logger.exception("bilateral default reach of %s could not be read", subject)
        return False


def states_answer(decision: ScopeDecision, spec: Spec, *, minted: bool, read_within: bool) -> bool:
    """Whether `decision` is the scope the finished answer `spec` states: decided for its
    spec_id, compiling the composite `spec` is where `spec` is minted as one (`minted`), and
    compiling none where it is not. A Spec whose rows were read within a composite's
    population (`read_within`) is stated only as that composite, so where it is not minted
    as one, no decision states it."""
    if decision.spec_id != spec.spec_id:
        return False
    if not minted:
        return not read_within and not decision.composite
    return (
        decision.composite
        and decision.scope is not None
        and decision.scope.compiled.spec_id == spec.spec_id
    )


def decide_after_read(
    spec: Spec,
    *,
    minted: bool,
    read_within: bool,
    selection: ScopeState,
    operations: Sequence[Operation],
    coverage: CoverageSnapshot,
) -> ScopeDecision | None:
    """The scope the finished answer `spec` states where no decision made before its read
    states it (`states_answer`). The reading was decided once, before the read, so this
    decision chooses nothing. It is decided from `spec` (`decide_spec_scope`) and kept only
    where it states `spec` as minted, the composite `spec` is where `minted`, no composite
    otherwise.

    A Spec whose rows were read within a composite's population (`read_within`) and that is
    not minted as that composite is a fault before anything is decided: no decision can label
    those rows, the table's would state the whole table and a composite's would not state the
    Spec. A decision that raises, or that states any other scope, is a fault too. A fault is
    logged at ERROR and noted as a degraded read, so no cache stores the answer, and the
    answer fails closed. A Spec minted as a composite, or read within a composite's
    population, has no table label to stand under, so the decision is None and the caller
    answers no_data `facet_not_available`. Any other answer reads as the one table it
    resolved and states no scope, saying it reads total ODA where the bilateral default
    reaches it (`undecided_choice`)."""
    if read_within and not minted:
        logger.error(
            "the finished answer %s was read within a composite's population but is not "
            "minted as the composite; it fails closed and is not cached",
            spec.spec_id,
        )
        note_degraded_read("scope decision")
        return None
    try:
        decision = decide_spec_scope(
            spec, selection=selection, operations=operations, coverage=coverage
        )
    except Exception as exc:
        logger.error(
            "scope decision for the finished answer %s raised; it fails closed and is not cached",
            spec.spec_id,
            exc_info=exc,
        )
    else:
        if states_answer(decision, spec, minted=minted, read_within=read_within):
            return decision
        logger.error(
            "the finished answer %s is %s, but its scope decided after the read %s; it fails "
            "closed and is not cached",
            spec.spec_id,
            "minted as a composite" if minted else "its table",
            "compiles the composite " + decision.scope.compiled.spec_id
            if decision.composite and decision.scope is not None
            else "compiles no composite",
        )
    note_degraded_read("scope decision")
    if minted or read_within:
        return None
    facts = answer_scope_facts(
        entity_groups={}, facets={}, sources=(), operations=operations, coverage=coverage
    )
    identity = spec.definition.identity
    choice = undecided_choice(identity, selection=selection, facts=facts, subject=spec.spec_id)
    return build_part_scope(identity, choice, spec_id=spec.spec_id, facts=facts)


def states_no_scope(identity: BaseIdentity, selection: ScopeState) -> bool:
    """Whether an answer of `identity` states no scope, and so no composite, whatever its
    facts (`decide_scope`): no binder settled an axis (`selection`) and the measure-basis
    axis does not apply to the identity (`basis.basis_dataset`). It reads no facts."""
    return not selection.records and basis_dataset(identity, get_index()) is None


def choose_provider(
    identity: BaseIdentity, selection: ScopeState, facts: ScopeFacts
) -> ProviderChoice:
    """The provider choice of a fresh reading of `identity` under the axis records and
    entries the pass's binders hold (`selection`), read in `facts` (`ProviderChoice`).

    DAC2a's total ODA is read at its bilateral destinations alone where the part names none
    or only bilateral ones (`provider.with_bilateral_default`), before any axis is read. A
    provider entry asking for total ODA withholds that default
    (`provider.bilateral_default_reading`). The provider record is settled over the
    identity so served, once words summing the bilateral and multilateral lines are read as
    its dataset sums them and words naming multilateral ODA are dropped where that dataset
    offers no such option. An answer whose default does not apply and whose binders settled
    no axis, on an identity the measure-basis axis does not apply to, states no scope, and
    nothing more is settled."""
    withheld = bilateral_default_withheld(identity, selection, facts)
    served, selection = bilateral_default_reading(identity, selection, facts)
    if served is identity and states_no_scope(identity, selection):
        return ProviderChoice(
            states_scope=False,
            bilateral=None,
            withheld=withheld,
            selection=selection,
            provider=None,
        )
    bilateral = (
        None
        if served is identity
        else composite_ref(served.model_dump(mode="json", include={"facets"}))
    )
    # Words summing the bilateral and multilateral lines ask for the dataset's own sum.
    selection = with_summed_provider(served, selection, facts)
    # "Multilateral ODA" names DAC1's row only on a dataset that publishes it.
    selection = without_unoffered_multilateral(served, selection, facts)
    return ProviderChoice(
        states_scope=True,
        bilateral=bilateral,
        withheld=withheld,
        selection=selection,
        provider=settle_provider(served, selection, facts),
    )


def build_part_scope(
    identity: BaseIdentity, choice: ProviderChoice, *, spec_id: str, facts: ScopeFacts
) -> ScopeDecision:
    """The scope the definite answer `spec_id` of `identity` states under the provider
    `choice` made for it (`choose_provider`), built over `facts`. It chooses nothing. The
    composite `choice.bilateral` names is bound to `identity` as the bilateral default binds
    it, and `choice.provider` is written as the provider record.

    The scope is built from the axis records the pass's binders hold
    (`choice.selection`) and the provider record. Only the entries a binder applied compile
    (`AxisRecord.entry`). The other entries that attach to the part (`ScopeState.attached`)
    are reported in `PartScope.dropped_entries`. An answer the recorded axes do not reach, as
    a region read at its published total, states none. A part whose binders settled no axis,
    such as one naming no place, states the scope of its identity where the measure-basis
    axis applies to it, with the stated or carried basis that identity holds applied
    (`basis.unbound_basis_record`). A fresh search does not refuse its scope
    (`search.scope.build.build_part_scope` with `refuse` false).

    Where the bilateral default applies, the scope's base is its composite, with or without
    a provider axis, and a part with no axis states a scope with no choice, whose compiled
    identity is the composite. A choice whose default could not apply carries the warning
    saying so (`bilateral_withheld_warning`)."""
    warning = None if choice.withheld is None else _withheld_warning(spec_id, choice.withheld)
    if not choice.states_scope:
        return ScopeDecision(
            spec_id=spec_id, facts=facts, scope=None, composite=False, warning=warning
        )
    if choice.bilateral is not None:
        identity = apply_binding(
            identity, ProviderBinding(scope="bilateral", provenance=choice.bilateral), []
        )
    selection = choice.selection
    records = dict(selection.records)
    if not records:
        # No binder settled an axis, as on a part naming no place. The basis axis applies
        # to the served identity, so the scope reads it and the part's entries.
        stated = selection.entries_on("measure_basis")
        if (basis := unbound_basis_record(identity, get_index(), stated)) is not None:
            records[basis.axis] = basis
    settled = ScopeState(entries=selection.entries, records=records)
    if choice.provider is not None:
        settled.write(choice.provider)
    built = scope_build.build_part_scope(
        base=identity,
        entries=[],
        records=settled.records,
        facts=facts,
        read=settled.attached(),
        refuse=False,
    )
    assert isinstance(built, PartScope)
    composite = (
        composite_ref(built.compiled.identity.model_dump(mode="json", include={"facets"}))
        is not None
    )
    if not built.choices and not composite:
        return ScopeDecision(
            spec_id=spec_id, facts=facts, scope=None, composite=False, warning=warning
        )
    return ScopeDecision(
        spec_id=spec_id, facts=facts, scope=built, composite=composite, warning=warning
    )


def decide_scope(
    spec_id: str, identity: BaseIdentity, *, selection: ScopeState, facts: ScopeFacts
) -> ScopeDecision:
    """The scope the definite answer `spec_id` of `identity` states: the provider choice
    made over `identity` itself (`choose_provider`), applied to it (`build_part_scope`).
    A fresh reading whose choice is made before its read applies it to the identity minted
    after it (`reading.elect_reading`)."""
    return build_part_scope(
        identity, choose_provider(identity, selection, facts), spec_id=spec_id, facts=facts
    )


READ_WINDOW_NOT_COVERED: WithheldWhy = "read_window_not_covered"
"""The `why` of an answer whose reading was elected over the years its read reads
(`reading.elect_reading`), where the bilateral destinations cover the range the answer
settled on but not every year read, so it reads total ODA."""

_WITHHELD_REASONS: Mapping[WithheldWhy, str] = {
    "destination_unclassified": (
        "the data holds a destination no side declaration places, so the bilateral "
        "destinations cannot be told apart"
    ),
    "option_not_covered": "the bilateral destinations do not cover every year of the answer",
    READ_WINDOW_NOT_COVERED: (
        "the bilateral destinations do not cover every year the request reads, though they "
        "cover the years shown"
    ),
    "scope_undecided": "the answer's scope could not be decided",
    "slot_not_served": (
        "the bilateral destinations cannot be read with what the part binds ({slot})"
    ),
}
"""The message's reason for each `why` (`provider.WithheldWhy`), `{slot}` the property of the
slot a `slot_not_served` reason names."""


def bilateral_withheld_warning(
    spec_id: str, identity: BaseIdentity, *, selection: ScopeState, facts: ScopeFacts
) -> Warning | None:
    """The info warning `bilateral_reading_unavailable` for the fresh reading `spec_id` of
    `identity`, stamped with that spec_id, where the bilateral default reaches it on a part
    with no provider axis and cannot apply (`provider.bilateral_default_withheld`), so its
    DAC2a figures are total ODA, core contributions to multilateral agencies included. Its
    params name why: the reason `provider.bilateral_default_withheld` gives. None
    otherwise."""
    withheld = bilateral_default_withheld(identity, selection, facts)
    return None if withheld is None else _withheld_warning(spec_id, withheld)


def _withheld_warning(spec_id: str, withheld: Withheld) -> Warning:
    """The `bilateral_reading_unavailable` warning for the reading `spec_id`, its params
    naming why the bilateral default is `withheld` (`Withheld.params`)."""
    return Warning(
        code=BILATERAL_READING_UNAVAILABLE,
        severity="info",
        message=(
            "These figures are total ODA, which includes donors' core contributions to "
            f"multilateral agencies: {_withheld_reason(withheld)}."
        ),
        spec_id=spec_id,
        params=withheld.params,
    )


def _withheld_reason(withheld: Withheld) -> str:
    """The message's words for `withheld` (`_WITHHELD_REASONS`), naming its slot."""
    return _WITHHELD_REASONS[withheld.why].format(slot=withheld.slot)
