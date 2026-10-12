"""A part's source scope as the region pipeline reads it.

A part's source mentions compile into a `SourceScope` before the pipeline runs
(`search.engine.source_scope`). This module holds what the pipeline reads from it. That is
the unbound-qualifier records that name the scope's sources in a refusal (`scope_records`),
the mark and warning of an excluding mention that removes nothing (`with_idle_exclusions`,
`idle_exclusion_warnings`) and of an including mention that serves nothing of the answer
(`with_idle_inclusions`, `idle_inclusion_warnings`), and whether the scope allows a
provenance of the UN System Data Commons (`allows_un_system`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Literal

from code_index.query import CodeIndex
from code_index.shared import get_index
from origin_catalogue import UN_SYSTEM, Catalogue, current_catalogue
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from search.engine.retrieval.bound_values import shape_datasets, shape_listed_rows
from search.engine.shape import ShapeDraft
from search.engine.source_scope import (
    NamedSource,
    SourceCollection,
    SourceScope,
    source_label,
)
from search.models import (
    GraphRef,
    IdentityBindingSet,
    IdentityBindingValue,
    ResolvedQualifier,
    SourceMention,
    Spec,
    Warning,
)


def source_record(ref: str) -> ResolvedQualifier:
    """The unbound record of the source `ref`, under its label (`source_label`)."""
    label = source_label(ref)
    return ResolvedQualifier(
        text=label, named=[GraphRef(dcid=ref, label=label)], slot=None, disposition="unbound"
    )


def _named_record(named: NamedSource) -> ResolvedQualifier:
    """The unbound record of a reference, or of a collection under its label with its
    members named."""
    members = named.members if isinstance(named, SourceCollection) else (named,)
    return ResolvedQualifier(
        text=named.label,
        named=[GraphRef(dcid=member.ref, label=member.label) for member in members],
        slot=None,
        disposition="unbound",
    )


def scope_records(scope: SourceScope) -> tuple[ResolvedQualifier, ...]:
    """The unbound records naming the sources `scope` was compiled from: those its part
    includes when it names any (`SourceScope.named`), else those it excludes, less a
    mention that resolved to no source or is marked idle (`SourceScope.idle`), which
    limited no answer."""
    polarity = "include" if scope.named else "exclude"
    return tuple(
        _named_record(named)
        for position, (mention, named) in enumerate(
            zip(scope.mentions, scope.references, strict=True)
        )
        if mention.polarity == polarity and named is not None and position not in scope.idle
    )


def with_idle_exclusions(
    scope: SourceScope | None, shapes: Sequence[ShapeDraft]
) -> SourceScope | None:
    """`scope` with each excluding mention that removes nothing from a part that recalled
    `shapes` marked idle (`SourceScope.idle`). One mark decides both the warning that the
    exclusion was not applied (`idle_exclusion_warnings`) and that no refusal names it
    (`scope_records`).

    A mention whose words resolved to no source stands for no provenance, so it always
    removes nothing. Any other mention removes nothing when the provenances it stands for
    (`SourceScope.expansions`) may publish none of `shapes` (`_may_publish`): candidacy
    then keeps every shape it would keep without the mention, and no read of them meets a
    facet of those provenances."""
    if scope is None:
        return None
    index = get_index()
    catalogue = current_catalogue()
    idle = frozenset(
        position
        for position, (mention, expansion) in enumerate(
            zip(scope.mentions, scope.expansions, strict=True)
        )
        if mention.polarity == "exclude"
        and not any(_may_publish(shape, expansion, index, catalogue) for shape in shapes)
    )
    return dataclasses.replace(scope, idle=idle)


def idle_exclusion_warnings(scope: SourceScope | None) -> list[Warning]:
    """One `query_qualifier_dropped` warning per excluding mention of `scope` marked idle
    (`with_idle_exclusions`), quoting the mention's words."""
    return [
        _dropped_warning(
            f'The requested exclusion "{words}" was not applied: no data this question '
            "reads comes from it."
        )
        for words in _idle_words(scope, "exclude")
    ]


def with_idle_inclusions(scope: SourceScope, served: frozenset[str]) -> SourceScope:
    """`scope` with each including mention none of whose provenances
    (`SourceScope.expansions`) is among `served`, the provenances the part's answer binds
    (`answer_provenances`), marked idle (`SourceScope.idle`). Such a mention served nothing
    of the answer, so the part warns that it was not served (`idle_inclusion_warnings`)."""
    idle = frozenset(
        position
        for position, (mention, expansion) in enumerate(
            zip(scope.mentions, scope.expansions, strict=True)
        )
        if mention.polarity == "include" and expansion.isdisjoint(served)
    )
    return dataclasses.replace(scope, idle=scope.idle | idle)


def idle_inclusion_warnings(scope: SourceScope | None) -> list[Warning]:
    """One `query_qualifier_dropped` warning per including mention of `scope` marked idle
    (`with_idle_inclusions`), quoting the mention's words."""
    return [
        _dropped_warning(
            f'The requested source "{words}" was not served: no data in this answer comes from it.'
        )
        for words in _idle_words(scope, "include")
    ]


def answer_provenances(specs: Sequence[Spec]) -> frozenset[str] | None:
    """The provenances the identities of `specs` bind as their source (`facets.provenance`,
    `spec_build.with_served_provenance`), or `None` when one binds none, so what served it
    is unknown."""
    served: set[str] = set()
    for spec in specs:
        binding = (spec.definition.identity.facets or {}).get("provenance")
        if isinstance(binding, IdentityBindingValue):
            served.add(binding.ref)
        elif isinstance(binding, IdentityBindingSet):
            served.update(binding.refs)
        else:
            return None
    return frozenset(served)


def _idle_words(scope: SourceScope | None, polarity: Literal["include", "exclude"]) -> list[str]:
    """The words of each mention of `scope` with `polarity` marked idle, in posted order."""
    if scope is None:
        return []
    return [
        _mention_words(mention, named)
        for position, (mention, named) in enumerate(
            zip(scope.mentions, scope.references, strict=True)
        )
        if position in scope.idle and mention.polarity == polarity
    ]


def _dropped_warning(message: str) -> Warning:
    return Warning(code=QUERY_QUALIFIER_DROPPED, severity="degraded", message=message)


def _may_publish(
    shape: ShapeDraft, provenances: frozenset[str], index: CodeIndex, catalogue: Catalogue
) -> bool:
    """Whether one of `provenances` may publish `shape`, read as candidacy reads a shape's
    publishers (`phase_rank_bind._in_scope`): the datasets publishing its family
    (`shape_datasets`), else the provenances holding the rows the index lists for its
    five-tuple (`shape_listed_rows`), else the provenances the catalogue knows to hold one
    of its members (`Catalogue.variable_provenances`). Where none of these is known, any
    provenance but a dataset the index declares may, since a declared dataset publishes
    only the index's rows."""
    if datasets := shape_datasets(shape, index):
        return not provenances.isdisjoint(datasets)
    if listed := shape_listed_rows(shape, index):
        return any(not provenances.isdisjoint(row.provenances) for row in listed)
    known = {
        row.id for sv in shape.sv_arc_facts or () for row in catalogue.variable_provenances(sv)
    }
    if known:
        return not provenances.isdisjoint(known)
    return not provenances <= index.datasets.keys()


def _mention_words(mention: SourceMention, named: NamedSource | None) -> str:
    """The words a mention was posted with, else the label of what its ref names."""
    if mention.text is not None or named is None:
        return mention.text or ""
    return named.label


def allows_un_system(scope: SourceScope | None) -> bool:
    """Whether a part under `scope` may answer from the UN System Data Commons, which holds
    when it has no scope or its scope allows a provenance of that origin."""
    if scope is None:
        return True
    catalogue = current_catalogue()
    return any(
        row is not None and row.data_commons == UN_SYSTEM
        for row in map(catalogue.provenance, scope.allowed)
    )
