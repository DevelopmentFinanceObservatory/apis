"""The scope a definite answer states, decided from its identity and the part's facts
before its preview launches.

The answer phase decides it once the answer's identity is known
(`spec_build.answer_identity`), and the candidates path from its survivor's Spec, so an
answer the binders settle as a composite reads no preview of the one table it resolved.
`pipeline._with_scope_choices` states the decision on the finished answer, and decides it
there for an answer neither path decided. Every decision over a finished answer's Spec is
guarded (`decide_guarded`)."""

from __future__ import annotations

import dataclasses
import functools
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
    ResolvedFacet,
    Spec,
    Warning,
)
from search.scope.basis import basis_dataset, unbound_basis_record
from search.scope.build import build_part_scope
from search.scope.provider import (
    CONTRIBUTIONS,
    bilateral_default_reaches,
    bilateral_default_reading,
    bilateral_default_withheld,
    settle_provider,
    with_summed_provider,
    without_unoffered_multilateral,
)
from search.scope.state import ScopeState
from search.scope.tables import PartFacts, ScopeFacts
from series_coverage.shared import CoverageSnapshot

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ScopeDecision:
    """The scope the answer `spec_id` states, built over `facts`; None when it states none.
    `composite` when its compiled identity is a declared composite (`composite_ref`), which
    the answer is then restated as (`scope.composite.assemble_composite`). `warning` is the
    answer's `bilateral_reading_unavailable` warning where it reads total ODA because the
    bilateral default could not apply (`bilateral_withheld_warning`)."""

    spec_id: str
    facts: ScopeFacts
    scope: PartScope | None
    composite: bool
    warning: Warning | None = None


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
    with `why` `scope_undecided` (`_undecided_warning`).

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
    except (CompositeNotServable, *LOST_READ_ERRORS):
        logger.warning(
            "scope decision for %s failed; the answer reads as one table that states no scope",
            spec_id,
            exc_info=True,
        )
    except Exception:
        logger.exception(
            "scope decision for %s raised; the answer reads as one table that states no scope "
            "and is not cached",
            spec_id,
        )
        note_degraded_read("scope decision")
    facts = answer_scope_facts(
        entity_groups={}, facets={}, sources=(), operations=operations, coverage=coverage
    )
    return ScopeDecision(
        spec_id=spec_id,
        facts=facts,
        scope=None,
        composite=False,
        warning=_undecided_warning(spec_id, identity, selection=selection, facts=facts),
    )


def _undecided_warning(
    spec_id: str, identity: BaseIdentity, *, selection: ScopeState, facts: ScopeFacts
) -> Warning | None:
    """The `bilateral_reading_unavailable` warning, `why` `scope_undecided`, for the answer
    `spec_id` of `identity` whose scope decision raised, where the bilateral default reaches
    `identity` (`provider.bilateral_default_reaches`) and no stated or carried provider entry
    asks for total ODA. None otherwise, and where the reach check raises too, which is
    logged."""
    stated = selection.entries_on("provider")
    if stated and stated[0].option_id == CONTRIBUTIONS:
        return None
    try:
        reaches = bilateral_default_reaches(identity, facts)
    except Exception:
        logger.exception("bilateral default reach of %s could not be read", spec_id)
        return None
    return _withheld_warning(spec_id, "scope_undecided") if reaches else None


def decide_guarded(
    spec: Spec,
    *,
    selection: ScopeState,
    operations: Sequence[Operation],
    coverage: CoverageSnapshot,
) -> ScopeDecision:
    """The scope the finished answer `spec` states (`decide_spec_scope`), guarded
    (`guarded_scope_decision`), so a decision that raises leaves `spec` as the one table it
    resolved."""
    return guarded_scope_decision(
        spec.spec_id,
        functools.partial(
            decide_spec_scope,
            spec,
            selection=selection,
            operations=operations,
            coverage=coverage,
        ),
        identity=spec.definition.identity,
        selection=selection,
        operations=operations,
        coverage=coverage,
    )


def states_no_scope(identity: BaseIdentity, selection: ScopeState) -> bool:
    """Whether an answer of `identity` states no scope, and so no composite, whatever its
    facts (`decide_scope`): no binder settled an axis (`selection`) and the measure-basis
    axis does not apply to the identity (`basis.basis_dataset`). It reads no facts."""
    return not selection.records and basis_dataset(identity, get_index()) is None


def decide_scope(
    spec_id: str, identity: BaseIdentity, *, selection: ScopeState, facts: ScopeFacts
) -> ScopeDecision:
    """The scope the definite answer `spec_id` of `identity` states, built from the axis
    records the pass's binders hold (`selection`) and the provider scope settled over the
    identity (`provider.settle_provider`), after words summing the bilateral and
    multilateral lines are read as the answer's dataset sums them
    (`provider.with_summed_provider`) and words naming multilateral ODA are dropped where
    that dataset offers no such option (`provider.without_unoffered_multilateral`). Only
    the entries a binder applied compile (`AxisRecord.entry`). The other entries that attach
    to the part (`ScopeState.attached`) are reported in `PartScope.dropped_entries`. An
    answer the recorded axes do not reach, as a region read at its published total, states
    none. A part whose binders settled no axis, such as one naming no place, states the
    scope of its identity where the measure-basis axis applies to it, with the stated or
    carried basis that identity holds applied (`basis.unbound_basis_record`). A fresh
    search does not refuse its scope (`build_part_scope` with `refuse` false).

    DAC2a's total ODA is read at its bilateral destinations alone where the part names none
    or only bilateral ones (`provider.with_bilateral_default`), before any axis is read, so
    the scope's base is that composite, with or without a provider axis. A part with no axis
    then states a scope with no choice, whose compiled identity is the composite. A provider
    entry asking for total ODA withholds that default (`provider.bilateral_default_reading`).
    A part with no provider axis whose default could not apply carries the warning saying
    so (`bilateral_withheld_warning`)."""
    warning = bilateral_withheld_warning(spec_id, identity, selection=selection, facts=facts)
    served, selection = bilateral_default_reading(identity, selection, facts)
    if served is identity and states_no_scope(identity, selection):
        return ScopeDecision(
            spec_id=spec_id, facts=facts, scope=None, composite=False, warning=warning
        )
    identity = served
    records = dict(selection.records)
    if not records:
        # No binder settled an axis, as on a part naming no place. The basis axis applies
        # to the served identity, so the scope reads it and the part's entries.
        stated = selection.entries_on("measure_basis")
        if (basis := unbound_basis_record(identity, get_index(), stated)) is not None:
            records[basis.axis] = basis
    # Words summing the bilateral and multilateral lines ask for the dataset's own sum.
    selection = with_summed_provider(identity, selection, facts)
    # "Multilateral ODA" names DAC1's row only on a dataset that publishes it.
    selection = without_unoffered_multilateral(identity, selection, facts)
    settled = ScopeState(entries=selection.entries, records=records)
    provider = settle_provider(identity, selection, facts)
    if provider is not None:
        settled.write(provider)
    built = build_part_scope(
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


_WITHHELD_REASONS: Mapping[str, str] = {
    "destination_unclassified": (
        "the data holds a destination no side declaration places, so the bilateral "
        "destinations cannot be told apart"
    ),
    "option_not_covered": "the bilateral destinations do not cover every year of the answer",
    "scope_undecided": "the answer's scope could not be decided",
}
"""The message's reason for each `why` with words of its own; any other `why` names the
slot the bilateral composite cannot serve."""


def bilateral_withheld_warning(
    spec_id: str, identity: BaseIdentity, *, selection: ScopeState, facts: ScopeFacts
) -> Warning | None:
    """The info warning `bilateral_reading_unavailable` for the fresh reading `spec_id` of
    `identity`, stamped with that spec_id, where the bilateral default reaches it on a part
    with no provider axis and cannot apply (`provider.bilateral_default_withheld`), so its
    DAC2a figures are total ODA, core contributions to multilateral agencies included. Its
    params name why: the reason `provider.bilateral_default_withheld` gives. None
    otherwise."""
    why = bilateral_default_withheld(identity, selection, facts)
    return None if why is None else _withheld_warning(spec_id, why)


def _withheld_warning(spec_id: str, why: str) -> Warning:
    """The `bilateral_reading_unavailable` warning for the reading `spec_id`, naming `why`."""
    return Warning(
        code=BILATERAL_READING_UNAVAILABLE,
        severity="info",
        message=(
            "These figures are total ODA, which includes donors' core contributions to "
            f"multilateral agencies: {_withheld_reason(why)}."
        ),
        spec_id=spec_id,
        params={"why": why},
    )


def _withheld_reason(why: str) -> str:
    """The message's words for `why` (`_WITHHELD_REASONS`), else the slot it names."""
    return _WITHHELD_REASONS.get(
        why, f"the bilateral destinations cannot be read with what the part binds ({why})"
    )
