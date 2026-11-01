"""The scope a definite answer states, decided from its identity and the part's facts
before its preview launches.

The answer phase decides it once the answer's identity is known
(`spec_build.answer_identity`), and the candidates path from its survivor's Spec, so an
answer the binders settle as a composite reads no preview of the one table it resolved.
`pipeline._with_scope_choices` states the decision on the finished answer, and decides it
there for an answer neither path decided."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence

from code_index.composites import CompositeNotServable, composite_ref
from code_index.shared import get_index
from search.engine.degraded_reads import LOST_READ_ERRORS, note_degraded_read
from search.engine.departures import datasets_served
from search.models import (
    BaseIdentity,
    EntityGroup,
    Operation,
    PartScope,
    ProvenanceRef,
    ResolvedFacet,
    Spec,
)
from search.scope.basis import basis_dataset, unbound_basis_record
from search.scope.build import build_part_scope
from search.scope.provider import (
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
    the answer is then restated as (`scope.composite.assemble_composite`)."""

    spec_id: str
    facts: ScopeFacts
    scope: PartScope | None
    composite: bool


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
    operations: Sequence[Operation],
    coverage: CoverageSnapshot,
) -> ScopeDecision:
    """The decision `decide` makes for the answer `spec_id`, or, when it raises, a decision
    that states no scope and no composite, so the answer reads as the one table it resolved.
    `decide` builds the facts it reads, so an error there is guarded too.

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
    return ScopeDecision(spec_id=spec_id, facts=facts, scope=None, composite=False)


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
    search does not refuse its scope (`build_part_scope` with `refuse` false)."""
    if states_no_scope(identity, selection):
        return ScopeDecision(spec_id=spec_id, facts=facts, scope=None, composite=False)
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
    if not built.choices:
        return ScopeDecision(spec_id=spec_id, facts=facts, scope=None, composite=False)
    composite = (
        composite_ref(built.compiled.identity.model_dump(mode="json", include={"facets"}))
        is not None
    )
    return ScopeDecision(spec_id=spec_id, facts=facts, scope=built, composite=composite)
