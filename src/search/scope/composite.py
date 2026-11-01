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
`Component` per component with the years it serves, and the facts of the file a mint of it
writes (`composite_download`). The table identity's observation preview reads one table, so the
restated answer carries none. The engine reads the composite's preview per component and
composes it here (`compose_observations`, `search.engine.region.composite_answer`).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
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
    DownloadFacts,
    DownloadLicence,
    Headline,
    HeadlineYear,
    ObservationSeries,
    ObservationsOmissionReason,
    PreviewOrder,
    ProvenanceRef,
    ResolvedFacet,
    ResolvedStatVars,
    SlotValue,
    Spec,
    SpecDefinition,
    SpecObservations,
    TimeWindow,
    omitted_observations,
)
from search.observations import provenance_refs
from search.render.labels import join_labels
from search.scope.tables import ScopeFacts
from series_coverage.coverage_types import BoundRoles, Role
from series_coverage.index import CoverageIndex, elected_year

_ROLES: Mapping[str, Role] = {"origin": "donor", "destination": "recipient"}
"""The coverage role each flow role of an identity's entities reads."""

_NO_POINTS: ObservationsOmissionReason = "observations_no_points"
"""The omission of a composite whose components read and found no point in their years."""


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


def compose_observations(
    served: Sequence[ServedComponent], reads: Sequence[SpecObservations | None]
) -> SpecObservations:
    """The composite's observation preview, composed from each component's own preview as
    the mint composes its frame (`materialise.composite`).

    `reads` holds each component's preview in `served`'s order, None for a component with
    no read (one serving no year, no definite answer, a failed resolve, or one past the
    deadline). Each component
    keeps the points of its series dated in a year it serves (`ServedComponent.served`) and
    its recipe's periods hold, so a spliced component read for `latest` keeps only its own
    periods, and a series left with no point is dropped. A kept series keeps its own StatVar
    and is a copy of the component's, so its private estimate marker survives. No zero rows
    are written for a zero-filling stacked component, since the composite's coverage counts
    none.

    `total_obs`, on each series and in sum, subtracts the points dropped: every point of a
    dropped series, and the visible points cut from a kept one. A truncated series shows only
    its latest points, so a cut among its hidden earlier points goes uncounted. The cut
    falls at the tail in practice, where a stacked composite's shared years end before a
    component's do. A spliced component read for a range holds only its periods' years, so
    nothing is cut from it. `truncated` is true when a kept component preview was, and
    `order` is the kept previews' when they agree, else None.

    With no series left, the preview is omitted: `observations_no_points` when every
    component serving a year read and found no point in the years it serves; else the
    first omission (in component order) of another reason a component's preview carries;
    else, when a component serving a year has no read, `observations_probe_failed`. A
    component serving no year is not read, so it bears on none of these."""
    series: list[ObservationSeries] = []
    total = 0
    truncated = False
    orders: set[PreviewOrder | None] = set()
    for item, read in zip(served, reads, strict=True):
        if read is None or read.omitted is not None:
            continue
        removed = 0
        kept_any = False
        for each in read.series:
            points = [point for point in each.points if _serves(item, point.date)]
            if not points:
                removed += each.total_obs
                continue
            cut = len(each.points) - len(points)
            removed += cut
            series.append(
                each.model_copy(update={"points": points, "total_obs": each.total_obs - cut})
            )
            kept_any = True
        if kept_any:
            total += max(read.total_obs - removed, 0)
            truncated = truncated or read.truncated
            orders.add(read.order)
    if series:
        order = next(iter(orders)) if len(orders) == 1 else None
        return SpecObservations(series=series, total_obs=total, truncated=truncated, order=order)
    return _composite_omission(
        [read for item, read in zip(served, reads, strict=True) if item.served]
    )


def _serves(item: ServedComponent, date: str) -> bool:
    """Whether a point of `item` dated `date` lies in a year it serves and its recipe's
    periods hold. A date with no leading year is kept, as `search.models.in_window` keeps
    it."""
    try:
        year = int(date[:4])
    except ValueError:
        return True
    return _serves_year(item, year)


def _serves_year(item: ServedComponent, year: int) -> bool:
    """Whether `item` serves `year` and its recipe's periods hold it."""
    periods = item.component.recipe.periods
    return year in item.served and (periods is None or periods.holds(year))


def _composite_omission(reads: Sequence[SpecObservations | None]) -> SpecObservations:
    """Why a composite whose components serving a year, read as `reads`, kept no series
    shows none (`compose_observations`)."""
    if all(
        read is not None and (read.omitted is None or read.omitted.reason == _NO_POINTS)
        for read in reads
    ):
        dispatched = sum(_dispatched(read) for read in reads if read is not None)
        return omitted_observations(
            _NO_POINTS,
            f"the composite's components read {dispatched} (stat_var, role_tuple) pairs and "
            "found no point in the years each component serves.",
            dispatched=dispatched,
            empty=dispatched,
            hard_failed=0,
            facet_rejected=0,
        )
    own = next(
        (
            read.omitted
            for read in reads
            if read is not None and read.omitted is not None and read.omitted.reason != _NO_POINTS
        ),
        None,
    )
    if own is not None:
        return SpecObservations(series=[], total_obs=0, truncated=False, omitted=own)
    return omitted_observations(
        "observations_probe_failed",
        "a component of the composite was not read; Spec.resolved.observations holds no series.",
    )


def _dispatched(read: SpecObservations) -> int:
    """The pairs a component's preview dispatched: its series, or the count its
    `observations_no_points` omission states."""
    if read.omitted is None:
        return len(read.series)
    return read.omitted.dispatched or 0


def assemble_composite(
    spec: Spec,
    identity: BaseIdentity,
    spec_id: str,
    *,
    served: Sequence[ServedComponent],
    facts: ScopeFacts,
) -> tuple[Spec, list[Component]]:
    """`spec`, the answer resolved for composite `identity`'s table identity
    (`code_index.composites.table_identity`), restated as the composite under `spec_id`, and
    the composite's components (`CompiledScope.components`).

    `served` holds `identity`'s components with the years each serves (`served_components`),
    which the composite's preview filters by too. The restated answer's definition is
    `identity`; its coverage counts every component's series in the years each serves, over
    the window those years span; its
    headline counts the latest of those years over every component serving it
    (`_headline`); its StatVar counts are the components' summed (`_stat_vars`); its
    provenance facet names the composite, with the composite's label and the account its
    component tables give; its sources are its component tables, whose labels its
    `published_tables` joins, and its measure label reads the declaration's `measure_noun`
    (`_measure_label`), so its sentence cites both tables and names the composite's measure;
    its concept label stays the table identity's answer's, which names no basis; its download
    facts are the composite file's (`composite_download`); and it carries no observation
    preview, since the table identity's reads one table. The engine reads the composite's
    preview per component (`search.engine.region.composite_answer`).
    Each component names its role, the table it reads, the years it serves, how many
    StatVars it serves and its own identity. Raises
    `code_index.composites.CompositeNotServable` as `component_identities` does."""
    document = identity.model_dump(mode="json", exclude_none=True)
    index = facts.index
    declaration = _declaration(index, document)
    years = sorted({year for item in served for year in item.served})
    observations = sum(item.years.counts.get(year, 0) for item in served for year in item.served)
    window = TimeWindow(start_year=years[0], end_year=years[-1]) if years else None
    sources = _component_sources(index, declaration)
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
            "download": composite_download(served, identity, facts=facts),
            "measure_label": _measure_label(declaration, spec.resolved.measure_label),
            "published_tables": join_labels([source.label for source in sources]),
            "trace": spec.resolved.trace.model_copy(update={"resolved_sources": sources}),
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


def composite_download(
    served: Sequence[ServedComponent], identity: BaseIdentity, *, facts: ScopeFacts
) -> DownloadFacts | None:
    """The facts of the default file a Datasets mint of composite `identity` writes, counted
    from `facts`' coverage snapshot in memory as `materialise.composite` composes the frame.

    Each component writes its members' series-years in the years it serves (`served`, as
    `ServedComponent.served` and its recipe's periods state them). A zero-filling component
    (`ComponentRecipe.zero_absent`) writes, for each member a mint of it serves, one row per
    cell, a (recipient, donor, year), that some component's series fills in the years it
    serves, its own rows and its zero rows alike. Series count the distinct (StatVar,
    recipient, donor) keys written, zero-filled ones included. The window spans the years
    written. The licence is the one every component table's source-index entry names, else
    None.

    The count reads no facet: it keeps a member's rows under a source, unit or method the
    mint's facet election leaves out, and counts the zero rows of such a member, and under a
    unit request it keeps rows a conversion would drop (`unit.conversion.counted_conversion`).
    So `rows` can overstate the file, and `rows_exact` is always false. `rows`, `series` and
    `window` are None where some component's members are not settled in memory
    (`ServedComponent.members`), as for a family of `observationAbout` members, whose rows
    only facet metadata counts, and where the coverage snapshot holds no row of some member
    (`CoverageIndex.covers`) of a component that writes its members' own rows, as
    `download._coverage_extent` returns None for a plan reading one, since the count would
    leave that member's rows out. A zero-filling component's members count one row per cell
    whether or not the snapshot holds them. None for a `latest` window, which no mint of a
    composite serves."""
    if identity.requested_window.kind == "latest":
        return None
    licence = _shared_licence(served)
    coverage = facts.coverage.index
    if any(
        item.members is None
        or not (
            item.component.recipe.zero_absent or all(coverage.covers(sv) for sv in item.members)
        )
        for item in served
    ):
        return DownloadFacts(rows=None, rows_exact=False, series=None, window=None, licence=licence)
    rows = series = 0
    years: set[int] = set()
    cells: set[tuple[str, str, int]] = set()
    for item in served:
        for (_variable, recipient, donor), held in _served_series(item, coverage):
            cells.update((recipient, donor, year) for year in held)
            if not item.component.recipe.zero_absent:
                rows += len(held)
                series += 1
                years.update(held)
    places = {(recipient, donor) for recipient, donor, _year in cells}
    for item in served:
        if item.component.recipe.zero_absent and cells:
            members = len(cast("frozenset[str]", item.members))
            rows += members * len(cells)
            series += members * len(places)
            years.update(year for _recipient, _donor, year in cells)
    return DownloadFacts(
        rows=rows,
        rows_exact=False,
        series=series,
        window=TimeWindow(start_year=min(years), end_year=max(years)) if years else None,
        licence=licence,
    )


def _served_series(
    item: ServedComponent, coverage: CoverageIndex
) -> Iterator[tuple[tuple[str, str, str], frozenset[int]]]:
    """Each series of `item`'s members under its identity's places, with the years of it
    that `item` serves (`_serves_year`), for a series holding one."""
    if item.members is None or not item.served:
        return
    found = coverage.series_years(
        variables=sorted(item.members),
        bound=_bound(item.component.identity),
        first_year=min(item.served),
        last_year=max(item.served),
    )
    if found is None:
        return
    epoch = coverage.horizon.epoch_year
    for key, bits in zip(found.keys(), found.year_bits(), strict=True):
        held = frozenset(
            year
            for offset in range(bits.bit_length())
            if bits >> offset & 1 and _serves_year(item, year := epoch + offset)
        )
        if held:
            yield key, held


def _shared_licence(served: Sequence[ServedComponent]) -> DownloadLicence | None:
    """The licence every component table of `served` names (`download.provenance_licence`),
    or None when they name different ones or one names none."""
    # download.py imports the unit conversion catalogue, which imports the graph client,
    # which imports scope.basis, which imports this module.
    from search.engine.retrieval.download import provenance_licence

    licences = [provenance_licence(item.component.recipe.dataset) for item in served]
    first = licences[0] if licences else None
    return first if all(licence == first for licence in licences) else None


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
