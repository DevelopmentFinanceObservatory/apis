"""Composite answers in Search: the years a composite's components cover, and an answer
restated as the composite it serves.

A composite identity binds a declared composite provenance (`code_index.composites`): the
provider-inclusive total of a donor's bilateral flows and its imputed multilateral share
(stacked), or the OECD headline, net flows to 2017 and grant equivalents from 2018 (spliced).
The Datasets mint composes its frame from one read per component, keeping the years every
stacked component holds rows in. Search reads the same years from the request's coverage
snapshot, in memory (`component_years`), so the window a scope option offers is the window the
mint serves.

`assemble_composite` restates a Search answer resolved for the composite's table identity
(`code_index.composites.table_identity`) as the composite: its identity and spec_id, its
coverage over the composite's years, its provenance facet naming the composite, and one
`Component` per component with the years it serves. The observation preview is left out, since
it reads one table.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from code_index.composites import (
    ComponentIdentity,
    CompositeNotServable,
    component_identities,
    table_identity,
)
from code_index.members import SourceNotInFamily, identity_family_query, index_family
from code_index.query import CodeIndex
from code_index.types import CompositeDeclaration
from search.models import (
    BaseIdentity,
    Component,
    CoverageExact,
    ResolvedFacet,
    Spec,
    SpecDefinition,
    TimeWindow,
    YearSpan,
)
from search.scope.tables import ScopeFacts
from series_coverage.coverage_types import BoundRoles, Role
from series_coverage.index import CoverageIndex, elected_year

_ROLES: Mapping[str, Role] = {"origin": "donor", "destination": "recipient"}
"""The coverage role each flow role of an identity's entities reads."""


@dataclass(frozen=True, slots=True)
class YearCounts:
    """How many series of one identity's family hold each year it asks for, a year none
    holds counted zero, and, for a `latest` window, the year a read of it elects (None when
    no series holds a year)."""

    counts: Mapping[int, int]
    latest: int | None = None

    @property
    def held(self) -> frozenset[int]:
        """The years some series holds."""
        return frozenset(year for year, count in self.counts.items() if count > 0)


def component_years(identity: Mapping[str, object], *, facts: ScopeFacts) -> YearCounts:
    """The series of the family `identity` reads (`code_index.members.index_family`) under
    the places its entities bind, counted per year of its window in `facts`' coverage
    snapshot: every year of a range, clipped to the years the snapshot represents, and every
    year up to its build year for an `all` or `latest` window. A `latest` window also elects
    the year a read of it serves (`series_coverage.index.elected_year`). Empty for a family
    the index does not decide, or one whose named dataset does not publish it. Read once per
    identity for one `facts`."""
    key = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return facts.remembered(key, lambda: _component_years(identity, facts=facts))


def _component_years(identity: Mapping[str, object], *, facts: ScopeFacts) -> YearCounts:
    index, coverage = facts.index, facts.coverage.index
    try:
        family = index_family(index, identity_family_query(index, identity))
    except SourceNotInFamily:
        family = None
    if family is None or not family.rows:
        return YearCounts(counts={})
    window = cast("Mapping[str, object]", identity.get("requested_window") or {"kind": "all"})
    first, last = _years_asked(window, coverage)
    if first > last:
        return YearCounts(counts={})
    counts = coverage.series_by_year(
        variables=[row.dcid for row in family.rows],
        bound=_bound(identity),
        first_year=first,
        last_year=last,
    )
    latest = elected_year(counts) if window.get("kind") == "latest" else None
    return YearCounts(counts=counts, latest=latest)


def _years_asked(window: Mapping[str, object], coverage: CoverageIndex) -> tuple[int, int]:
    """The first and last year `window` asks for, inside the years `coverage` represents and
    not after its build year."""
    horizon = coverage.horizon
    last = horizon.max_year
    if coverage.build_year is not None:
        last = min(last, coverage.build_year)
    first = horizon.epoch_year
    if window.get("kind") == "range":
        start = cast("int | None", window.get("start_year"))
        end = cast("int | None", window.get("end_year"))
        first = first if start is None else max(first, start)
        last = last if end is None else min(last, end)
    return first, last


def _bound(identity: Mapping[str, object]) -> BoundRoles:
    """The places `identity` binds on each coverage role."""
    entities = cast("Mapping[str, Sequence[str]]", identity.get("entities") or {})
    return {
        _ROLES[role]: frozenset(dcids)
        for role, dcids in entities.items()
        if role in _ROLES and dcids
    }


def table_document(identity: BaseIdentity, index: CodeIndex) -> dict[str, object] | None:
    """`identity` as the document a reader of one table's family reads
    (`code_index.composites.table_identity`): itself, or the identity a composite's option was
    set on. None for a composite whose components cannot be derived."""
    try:
        return dict(table_identity(index, identity.model_dump(mode="json", exclude_none=True)))
    except CompositeNotServable:
        return None


def spans(years: Iterable[int]) -> list[YearSpan]:
    """`years` as closed runs of consecutive years, in order."""
    runs: list[YearSpan] = []
    for year in sorted(set(years)):
        if runs and runs[-1][1] == year - 1:
            runs[-1] = (runs[-1][0], year)
        else:
            runs.append((year, year))
    return runs


@dataclass(frozen=True, slots=True)
class _Served:
    component: ComponentIdentity
    years: YearCounts
    served: frozenset[int]


def _served(identity: Mapping[str, object], *, facts: ScopeFacts) -> list[_Served]:
    """Each component of composite `identity` with the years it serves: for a stacked
    composite the years every component holds a series in, for a spliced one the years its
    own component holds. Raises `code_index.composites.CompositeNotServable` as
    `component_identities` does, and `ValueError` for an identity that binds no composite."""
    components = component_identities(facts.index, identity)
    if components is None:
        raise ValueError("the identity binds no composite provenance")
    counted = [
        (component, component_years(component.identity, facts=facts)) for component in components
    ]
    if _declaration(facts.index, identity).kind == "stacked":
        shared = frozenset.intersection(*(years.held for _, years in counted))
        return [_Served(c, years, shared) for c, years in counted]
    return [_Served(c, years, years.held) for c, years in counted]


def _declaration(index: CodeIndex, identity: Mapping[str, object]) -> CompositeDeclaration:
    """The composite `identity`'s provenance facet binds, which `index` serves."""
    facets = cast("Mapping[str, Mapping[str, object]]", identity.get("facets") or {})
    ref = cast(str, facets["provenance"]["ref"])
    declaration = index.composite(ref)
    assert declaration is not None
    return declaration


def assemble_composite(
    spec: Spec, identity: BaseIdentity, spec_id: str, *, facts: ScopeFacts
) -> tuple[Spec, list[Component]]:
    """`spec`, the answer resolved for composite `identity`'s table identity
    (`code_index.composites.table_identity`), restated as the composite under `spec_id`, and
    the composite's components (`CompiledScope.components`).

    The restated answer's definition is `identity`; its coverage counts every component's
    series in the years each serves (`_served`), over the window those years span; its
    provenance facet names the composite, with the composite's label and the account its
    component tables give; and it carries no observation preview or one-year headline, which
    read one table. Each component names its role, the table it reads, the years it serves
    and its own identity. Raises `code_index.composites.CompositeNotServable` as
    `component_identities` does."""
    document = identity.model_dump(mode="json", exclude_none=True)
    index = facts.index
    served = _served(document, facts=facts)
    declaration = _declaration(index, document)
    years = sorted({year for item in served for year in item.served})
    observations = sum(item.years.counts.get(year, 0) for item in served for year in item.served)
    window = TimeWindow(start_year=years[0], end_year=years[-1]) if years else None
    accounts = {index.datasets[recipe.dataset].account for recipe in declaration.components}
    facets = {
        **spec.resolved.facets,
        "provenance": ResolvedFacet(
            basis="bound",
            ref=declaration.provenance,
            label=declaration.label,
            account=accounts.pop() if len(accounts) == 1 else None,
        ),
    }
    resolved = spec.resolved.model_copy(
        update={
            "coverage": CoverageExact(
                has_data=observations > 0, observation_count=observations, window=window
            ),
            "facets": facets,
            "observations": None,
            "headline": None,
        }
    )
    restated = spec.model_copy(
        update={
            "spec_id": spec_id,
            "definition": SpecDefinition(spec_id=spec_id, identity=identity),
            "resolved": resolved,
        }
    )
    return restated, _components(served)


def composite_components(identity: BaseIdentity, *, facts: ScopeFacts) -> list[Component] | None:
    """The components of composite `identity` (`CompiledScope.components`), each with its
    role, the table it reads, the years it serves and its own identity; None for an identity
    that binds no composite. Raises `code_index.composites.CompositeNotServable` as
    `component_identities` does."""
    document = identity.model_dump(mode="json", exclude_none=True)
    if component_identities(facts.index, document) is None:
        return None
    return _components(_served(document, facts=facts))


def _components(served: Sequence[_Served]) -> list[Component]:
    return [
        Component(
            role=item.component.role,
            periods=spans(item.served),
            provenance=item.component.recipe.dataset,
            identity=BaseIdentity.model_validate(item.component.identity),
        )
        for item in served
    ]
