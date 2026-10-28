"""Composite answers in Search: the years a composite's components cover, and an answer
restated as the composite it serves.

A composite identity binds a declared composite provenance (`code_index.composites`): the
provider-inclusive total of a donor's bilateral flows and its imputed multilateral share
(stacked), or the OECD headline, net flows to 2017 and grant equivalents from 2018 (spliced).
The Datasets mint composes its frame from one read per component, keeping the years every
stacked component's table covers (`materialise.composite.table_years`, `stacked_years`). Search
reads the same years from the request's coverage snapshot, in memory, and the years each
component holds under the part's places (`component_years`), so the window a scope option
offers is the window the mint serves.

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
    provenance_account,
    table_identity,
)
from code_index.members import SourceNotInFamily, identity_family_query, index_family
from code_index.query import CodeIndex
from code_index.types import CompositeDeclaration
from materialise.composite import spans, stacked_years, table_years
from search.engine.retrieval.bound_values import served_members
from search.models import (
    BaseIdentity,
    BindingValue,
    Component,
    CoverageExact,
    DateRange,
    Headline,
    HeadlineYear,
    ProvenanceRef,
    ResolvedFacet,
    ResolvedStatVars,
    SlotValue,
    Spec,
    SpecDefinition,
    TimeWindow,
)
from search.observations import provenance_refs
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


def donor_years(
    identity: Mapping[str, object], *, facts: ScopeFacts
) -> Mapping[str, frozenset[int]]:
    """The years of its window in which each donor `identity` names holds a series of the
    family it reads, as `component_years` counts them, by donor, from one scan of the
    family's rows in `facts`' coverage snapshot. A donor holding no series is absent. Read
    once per identity for one `facts`."""
    key = "donor_years:" + json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return facts.remembered(key, lambda: _donor_years(identity, facts=facts))


def _donor_years(
    identity: Mapping[str, object], *, facts: ScopeFacts
) -> Mapping[str, frozenset[int]]:
    index, coverage = facts.index, facts.coverage.index
    try:
        family = index_family(index, identity_family_query(index, identity))
    except SourceNotInFamily:
        family = None
    if family is None or not family.rows:
        return {}
    window = cast("Mapping[str, object]", identity.get("requested_window") or {"kind": "all"})
    first, last = _years_asked(window, coverage)
    if first > last:
        return {}
    found = coverage.series_years(
        variables=[row.dcid for row in family.rows],
        bound=_bound(identity),
        first_year=first,
        last_year=last,
    )
    if found is None:
        return {}
    epoch = coverage.horizon.epoch_year
    held: dict[str, set[int]] = {}
    for (_variable, _recipient, donor), bits in zip(found.keys(), found.year_bits(), strict=True):
        held.setdefault(donor, set()).update(
            epoch + offset for offset in range(bits.bit_length()) if bits >> offset & 1
        )
    return {donor: frozenset(years) for donor, years in held.items()}


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


@dataclass(frozen=True, slots=True)
class ServedComponent:
    """One component, the series its family holds per year under the part's places, the
    years it serves in the composite, and the StatVars a mint of it serves (None when the
    index does not settle them)."""

    component: ComponentIdentity
    years: YearCounts
    served: frozenset[int]
    members: frozenset[str] | None


def served_components(
    identity: Mapping[str, object], *, facts: ScopeFacts
) -> list[ServedComponent]:
    """Each component of composite `identity` with the years it serves, as the mint composes
    them (`materialise.composite`). A stacked composite's years are those every component's
    table covers (`stacked_years`); in them, a component serves the years it holds a series
    in, and a zero-filling one (`ComponentRecipe.zero_absent`) every year any component holds
    one in, since its zero rows fill the cells it lacks. A spliced component serves the years
    it holds. Raises `code_index.composites.CompositeNotServable` as `component_identities`
    does, and `ValueError` for an identity that binds no composite."""
    components = component_identities(facts.index, identity)
    if components is None:
        raise ValueError("the identity binds no composite provenance")
    counted = [
        (component, component_years(component.identity, facts=facts)) for component in components
    ]
    if _declaration(facts.index, identity).kind != "stacked":
        return [ServedComponent(c, years, years.held, _members(c)) for c, years in counted]
    covered = stacked_years(_table_years(c.identity, facts=facts) for c, _ in counted)
    held_by_any = frozenset[int]().union(*(years.held for _, years in counted))
    return [
        ServedComponent(
            c,
            years,
            covered & (held_by_any if c.recipe.zero_absent else years.held),
            _members(c),
        )
        for c, years in counted
    ]


def _table_years(identity: Mapping[str, object], *, facts: ScopeFacts) -> frozenset[int]:
    """The years the table component `identity` reads covers in `facts`' coverage snapshot
    (`materialise.composite.table_years`), read once per identity for one `facts`."""
    key = "table:" + json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return facts.remembered(key, lambda: table_years(facts.index, facts.coverage.index, identity))


def _members(component: ComponentIdentity) -> frozenset[str] | None:
    """The StatVars a Datasets mint of `component` serves (`bound_values.served_members`)."""
    return served_members(BaseIdentity.model_validate(component.identity))


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
    series in the years each serves (`served_components`), over the window those years span; its
    headline counts the latest of those years over every component serving it
    (`_headline`); its StatVar counts are the components' summed (`_stat_vars`); its
    provenance facet names the composite, with the composite's label and the account its
    component tables give; its sources are its component tables and its measure label reads
    the declaration's `measure_noun` (`_measure_label`), so its sentence cites both tables
    and names the composite's measure; and it carries no observation preview, which reads
    one table.
    Each component names its role, the table it reads, the years it serves, how many
    StatVars it serves and its own identity. Raises
    `code_index.composites.CompositeNotServable` as `component_identities` does."""
    document = identity.model_dump(mode="json", exclude_none=True)
    index = facts.index
    served = served_components(document, facts=facts)
    declaration = _declaration(index, document)
    years = sorted({year for item in served for year in item.served})
    observations = sum(item.years.counts.get(year, 0) for item in served for year in item.served)
    window = TimeWindow(start_year=years[0], end_year=years[-1]) if years else None
    facets = {
        **spec.resolved.facets,
        "provenance": ResolvedFacet(
            basis="bound",
            ref=declaration.provenance,
            label=declaration.label,
            account=provenance_account(index, declaration.provenance),
        ),
    }
    resolved = spec.resolved.model_copy(
        update={
            "coverage": CoverageExact(
                has_data=observations > 0, observation_count=observations, window=window
            ),
            "facets": facets,
            "observations": None,
            "headline": _headline(served, document, spec.resolved.headline, facts=facts),
            "stat_vars": _stat_vars(served, spec.resolved.stat_vars, document, facts=facts),
            "slots": [
                slot.model_copy(update={"binding": _window_binding(window)})
                if slot.key.axis == "when" and window is not None
                else slot
                for slot in spec.resolved.slots
            ],
            "download": None,
            "measure_label": _measure_label(declaration, spec.resolved.measure_label),
            "trace": spec.resolved.trace.model_copy(
                update={"resolved_sources": _component_sources(index, declaration)}
            ),
        }
    ).stating_data_years((years[0], years[-1]) if years else None)
    restated = spec.model_copy(
        update={
            "spec_id": spec_id,
            "definition": SpecDefinition(spec_id=spec_id, identity=identity),
            "resolved": resolved,
        }
    )
    return restated, _components(served)


def _measure_label(declaration: CompositeDeclaration, measure: str | None) -> str | None:
    """The measure label a composite answer's sentence names: a stacked composite's
    `measure_noun` after `measure`, the label of the measure its components share, and a
    spliced one's in its place. A stacked composite with no measure label keeps none, so the
    sentence names its members' measure as the table identity's would."""
    if declaration.kind == "spliced":
        return declaration.measure_noun
    return f"{measure} {declaration.measure_noun}" if measure is not None else None


def _component_sources(index: CodeIndex, declaration: CompositeDeclaration) -> list[ProvenanceRef]:
    """The tables a composite is served from, in component order, once each, as the
    provenance refs a sentence's byline and a citation name (`observations.provenance_refs`)."""
    datasets = [recipe.dataset for recipe in declaration.components]
    return provenance_refs(
        datasets, {dataset: index.datasets[dataset].label for dataset in datasets}
    )


def _window_binding(window: TimeWindow) -> BindingValue:
    """The when slot's binding to `window`, as `spec_build.bind_when_slot` binds it."""
    return BindingValue(value=SlotValue(ref=None, value_kind="time_window", time_window=window))


def composite_components(identity: BaseIdentity, *, facts: ScopeFacts) -> list[Component] | None:
    """The components of composite `identity` (`CompiledScope.components`), each with its
    role, the table it reads, the years it serves and its own identity; None for an identity
    that binds no composite. Raises `code_index.composites.CompositeNotServable` as
    `component_identities` does."""
    document = identity.model_dump(mode="json", exclude_none=True)
    if component_identities(facts.index, document) is None:
        return None
    return _components(served_components(document, facts=facts))


def _components(served: Sequence[ServedComponent]) -> list[Component]:
    return [
        Component(
            role=item.component.role,
            periods=list(spans(item.served)),
            provenance=item.component.recipe.dataset,
            stat_vars=len(item.members) if item.members is not None else None,
            identity=BaseIdentity.model_validate(item.component.identity),
        )
        for item in served
    ]


def _with_data(
    item: ServedComponent, bound: BoundRoles, coverage: CoverageIndex, years: Iterable[int]
) -> frozenset[str]:
    """The StatVars of `item` holding a series under `bound` in one of `years`."""
    held = sorted(set(years))
    if item.members is None or not held:
        return frozenset()
    found = coverage.series_years(
        variables=sorted(item.members), bound=bound, first_year=held[0], last_year=held[-1]
    )
    if found is None:
        return frozenset()
    series = found.keys()
    return frozenset(variable for variable, _, _ in series)


def _stat_vars(
    served: Sequence[ServedComponent],
    stated: ResolvedStatVars,
    identity: Mapping[str, object],
    *,
    facts: ScopeFacts,
) -> ResolvedStatVars:
    """`stated`, the table identity's StatVar counts, restated over every component. The
    member count is the StatVars holding a series in the years their component serves, the
    served count sums the StatVars a mint of each component serves (None where the index does
    not settle some component's members), and the served date range spans the years the
    components serve."""
    years = sorted({year for item in served for year in item.served})
    bound = _bound(identity)
    confirmed = sum(
        len(_with_data(item, bound, facts.coverage.index, item.served)) for item in served
    )
    counts = [len(item.members) if item.members is not None else None for item in served]
    return stated.model_copy(
        update={
            "member_count": confirmed,
            "member_count_exact": True,
            "served_count": None if None in counts else sum(c for c in counts if c is not None),
            "data_date_range": (
                DateRange(start=str(years[0]), end=str(years[-1])) if years else None
            ),
        }
    )


def _headline(
    served: Sequence[ServedComponent],
    identity: Mapping[str, object],
    stated: Headline | None,
    *,
    facts: ScopeFacts,
) -> Headline | None:
    """The composite's headline: the latest year any component serves, counted over the
    rows of every component serving it (`CoverageIndex.headline` over their StatVars, one
    year at a time), with the same counts for it and the two years before it among those
    the composite serves. The named window is `stated`'s, the request's. None when no
    component serves a year or settles its members."""
    coverage = facts.coverage.index
    bound = _bound(identity)
    years = sorted({year for item in served for year in item.served}, reverse=True)[:3]
    counted: list[HeadlineYear] = []
    for year in years:
        variables = sorted(
            {
                variable
                for item in served
                if year in item.served and item.members is not None
                for variable in item.members
            }
        )
        found = (
            coverage.headline(variables=variables, bound=bound, first_year=year, last_year=year)
            if variables
            else None
        )
        if found is None or found.year != year:
            continue
        names = coverage.role_names.get(variables[0])
        role_dcids = {} if names is None else {"donor": names.donor, "recipient": names.recipient}
        counted.append(
            HeadlineYear(
                year=year,
                observations=found.observations,
                series=found.series,
                role_counts={
                    dcid: count
                    for role, count in found.role_counts.items()
                    if (dcid := role_dcids.get(role)) is not None
                },
            )
        )
    if not counted:
        return None
    head = counted[0]
    return Headline(
        year=head.year,
        year_named=stated.year_named if stated is not None else False,
        named_window=stated.named_window if stated is not None else None,
        observations=head.observations,
        series=head.series,
        role_counts=head.role_counts,
        recent_years=counted,
    )
