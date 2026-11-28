"""Composite answers in Search: the years a composite's components cover, and the Spec minted
as the composite it serves.

A composite identity binds a declared composite provenance (`code_index.composites`): the
provider-inclusive total of a donor's bilateral flows and its imputed multilateral share
(stacked), or the OECD headline, net flows to 2017 and grant equivalents from 2018 (spliced).
The Datasets mint composes its frame from one read per component, keeping the years a stacked
composite serves for the frame's places (`materialise.composite.stacked_years`). Search states
those years by the same rule from the request's coverage snapshot, in memory
(`materialise.composite.read_component_years`), and counts the series each component holds
under the part's places (`component_years`). The mint reads a component's held years from the
rows it read, so a year a source publishes before the snapshot rebuilds can appear in the mint
before Search states it.

`CompositeMint` mints a Search answer's Spec as the composite (`spec_build.build_spec(composite=)`)
over the table identity's arguments (`code_index.composites.table_identity`): its identity and
spec_id, its coverage over the composite's years, its headline, its open role axes, its StatVar
counts, its provenance facet naming the composite, its download facts (`composite_download`), and
one `Component` per component with the years it serves. The table identity's observation preview
reads one table, so the minted Spec carries none. The engine reads the composite's preview per
component and composes it (`compose_observations`, `search.engine.region.composite_answer`). One
method derives each field.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from typing import cast

from code_index.composites import (
    ComponentIdentity,
    CompositeNotServable,
    component_identities,
    components_splice,
    components_sum,
    provenance_account,
    table_identity,
)
from code_index.members import pinned_refs
from code_index.query import CodeIndex
from code_index.scope_conventions import RolePopulation
from code_index.types import CompositeDeclaration
from materialise.composite import (
    ComponentYears,
    bound_roles,
    component_reads,
    family_variables,
    read_component_years,
    spans,
    stacked_years,
    window_include,
)
from materialise.enrichment import PartySplit, party_split
from origin_catalogue import Catalogue, bound_catalogue, pinned_reads, scoped_reads
from search.engine.identity_edit import open_within
from search.engine.retrieval.bound_values import served_members
from search.engine.source_scope import read_provenances
from search.engine.spec_id import identity_spec_id
from search.models import (
    BaseIdentity,
    BindingValue,
    Component,
    CoverageExact,
    DateRange,
    DownloadFacts,
    DownloadLicence,
    EntityRoleKey,
    Headline,
    HeadlineYear,
    ObservationSeries,
    ObservationsOmissionReason,
    PreviewOrder,
    ProvenanceRef,
    ResolvedEntity,
    ResolvedFacet,
    ResolvedScope,
    ResolvedStatVars,
    RoleAxis,
    RoleAxisFact,
    Slot,
    SlotValue,
    Spec,
    SpecObservations,
    TimeWindow,
    Warning,
    omitted_observations,
)
from search.observations import composite_facet_ref, provenance_refs
from search.render.labels import join_labels
from search.scope.tables import ScopeFacts
from series_coverage.coverage_types import AXIS_COVERAGE_ROLES, BoundRoles, Role
from series_coverage.coverage_types import Headline as CoverageHeadline
from series_coverage.index import CoverageIndex, elected_year
from series_coverage.schema import ABSENT_ENTITY

logger = logging.getLogger(__name__)

_NO_POINTS: ObservationsOmissionReason = "observations_no_points"
"""The omission of a composite whose components read and found no point in their years."""


@dataclass(frozen=True, slots=True)
class YearCounts:
    """How many series of one identity's family hold each year it asks for, a year none
    holds counted zero, and, for a `latest` window, the year a read of it elects (None when
    no series holds a year). `settled` when the index decides the family, the window falls
    within the snapshot's years and the snapshot holds a row of each of the family's
    StatVars, so a year counted zero is one no series holds; otherwise an empty count may
    only be the snapshot not knowing."""

    counts: Mapping[int, int]
    latest: int | None = None
    settled: bool = False

    @property
    def held(self) -> frozenset[int]:
        """The years some series holds."""
        return frozenset(year for year, count in self.counts.items() if count > 0)


def component_years(identity: Mapping[str, object], *, facts: ScopeFacts) -> YearCounts:
    """The series of the family `identity` reads (`code_index.members.index_family`) under
    the places its entities bind, counted per year of its window in `facts`' coverage
    snapshot: every year of a range, clipped to the years the snapshot represents, and every
    year up to its build year for an `all` or `latest` window, less the held-back years its
    window does not opt in to (`window_include`), as a mint of it cuts them. A `latest`
    window also elects the year a read of it serves (`series_coverage.index.elected_year`).
    Empty for a family
    the index does not decide, or one whose named dataset does not publish it. An identity
    that pins provenances reads under them, within the caller's source scope
    (`_pinned_reads`). A composite's component read clears that scope first
    (`origin_catalogue.pinned_reads`), as a mint of it reads its rows. Read once per
    identity and source scope for one `facts`."""
    key = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return facts.remembered(key, lambda: _component_years(identity, facts=facts))


def _component_years(identity: Mapping[str, object], *, facts: ScopeFacts) -> YearCounts:
    with _pinned_reads(identity, facts.index):
        return _counted_years(identity, facts=facts)


def _counted_years(identity: Mapping[str, object], *, facts: ScopeFacts) -> YearCounts:
    coverage = facts.coverage.index
    variables = family_variables(facts.index, identity)
    if not variables:
        return YearCounts(counts={})
    window = cast("Mapping[str, object]", identity.get("requested_window") or {"kind": "all"})
    first, last = _years_asked(window, coverage)
    if first > last:
        return YearCounts(counts={})
    counts = coverage.series_by_year(
        variables=variables,
        bound=bound_roles(identity),
        first_year=first,
        last_year=last,
        include=window_include(identity),
    )
    latest = elected_year(counts) if window.get("kind") == "latest" else None
    settled = all(coverage.covers(variable) for variable in variables)
    return YearCounts(counts=counts, latest=latest, settled=settled)


def donor_years(
    identity: Mapping[str, object], *, facts: ScopeFacts
) -> Mapping[str, frozenset[int]]:
    """The years of its window in which each donor `identity` names holds a series of the
    family it reads, as `component_years` counts them, by donor, from one scan of the
    family's rows in `facts`' coverage snapshot, under the provenances `identity` pins
    within the caller's source scope where it pins any (`_pinned_reads`). A donor holding
    no series is absent. Read once per identity and source scope for one `facts`."""
    key = "donor_years:" + json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return facts.remembered(key, lambda: _donor_years(identity, facts=facts))


def _donor_years(
    identity: Mapping[str, object], *, facts: ScopeFacts
) -> Mapping[str, frozenset[int]]:
    with _pinned_reads(identity, facts.index):
        return _held_by_donor(identity, facts=facts)


def _pinned_reads(
    identity: Mapping[str, object], index: CodeIndex
) -> AbstractContextManager[object]:
    """Binds the caller's catalogue narrowed to the provenances `identity` pins, within its
    source scope (`origin_catalogue.scoped_reads`), as a plain pinned table reads under a
    part's scope. It binds nothing where `identity` pins none. A composite's component reads run under
    `origin_catalogue.pinned_reads(None)` first, so there the pins replace the part's
    scope."""
    return scoped_reads(_pins(identity, index))


def _held_by_donor(
    identity: Mapping[str, object], *, facts: ScopeFacts
) -> Mapping[str, frozenset[int]]:
    coverage = facts.coverage.index
    variables = family_variables(facts.index, identity)
    if not variables:
        return {}
    window = cast("Mapping[str, object]", identity.get("requested_window") or {"kind": "all"})
    first, last = _years_asked(window, coverage)
    if first > last:
        return {}
    found = coverage.series_years(
        variables=variables,
        bound=bound_roles(identity),
        first_year=first,
        last_year=last,
        include=window_include(identity),
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


def table_document(identity: BaseIdentity, index: CodeIndex) -> dict[str, object] | None:
    """`identity` as the document a reader of one table's family reads
    (`code_index.composites.table_identity`): itself, or the identity a composite's option was
    set on. None for a composite whose components cannot be derived."""
    try:
        return dict(
            table_identity(index, identity.model_dump(mode="json", exclude_none=True)).identity
        )
    except CompositeNotServable:
        return None


@dataclass(frozen=True, slots=True)
class ServedComponent:
    """One component, the series its family holds per year under the part's places, the
    years it serves in the composite, all within its recipe's periods, and the StatVars a
    mint of it serves (None when the index does not settle them). `elected` is the year a
    composite under a `latest` window elects, the latest year any of its components serves,
    and None for any other window."""

    component: ComponentIdentity
    years: YearCounts
    served: frozenset[int]
    members: frozenset[str] | None
    elected: int | None = None

    @property
    def preview_years(self) -> frozenset[int]:
        """The years the component's preview reads rows in, the years it serves or,
        under a `latest` window, the year the composite elects (`elected`) alone where it is
        one of them. Empty when it previews no year, and then it is not read."""
        return self.served if self.elected is None else self.served & {self.elected}


def served_components(
    identity: Mapping[str, object], *, facts: ScopeFacts
) -> list[ServedComponent]:
    """Each component of composite `identity` with the years it serves, as the mint composes
    them (`materialise.composite`). A stacked composite's years follow the mint's rule
    (`stacked_years`) over the coverage snapshot: every component's table covers them, and
    every component that reads a missing row as unknown holds a series in them under some of
    its places. The mint reads those held years from its rows, so it can serve a year
    before the snapshot holds it. Every component reads with no source scope but the
    provenances its identity pins (`origin_catalogue.pinned_reads`), whatever scope the caller reads under,
    so a part's scope build and its composite answer
    (`region.composite_answer.composite_served`) read the tables the mint reads. In them, a
    component serves the years it holds a series in, and a zero-filling one
    (`ComponentRecipe.zero_absent`) every year any component holds one in, since its zero
    rows fill the cells it lacks. A spliced component serves the years it holds within its
    recipe's periods, so under a `latest` window, which its identity passes unclipped
    (`code_index.composites.component_identities`), it serves no year after its period
    ends. Under a `latest` window either kind elects the latest year any component serves
    (`ServedComponent.elected`), and a member holding no data in that year serves nothing.
    Raises `code_index.composites.CompositeNotServable` as
    `component_identities` does, and `ValueError` for an identity that binds no
    composite."""
    components = component_identities(facts.index, identity)
    if components is None:
        raise ValueError("the identity binds no composite provenance")
    with pinned_reads(None):
        return _served(identity, components, facts=facts)


def _served(
    identity: Mapping[str, object], components: Sequence[ComponentIdentity], *, facts: ScopeFacts
) -> list[ServedComponent]:
    counted = [
        (component, component_years(component.identity, facts=facts)) for component in components
    ]
    if _declaration(facts.index, identity).kind == "stacked":
        covered = stacked_years(_read_years(c, facts=facts) for c, _ in counted)
        held_by_any = frozenset[int]().union(*(years.held for _, years in counted))
        served = [
            (c, years, covered & (held_by_any if c.recipe.zero_absent else years.held))
            for c, years in counted
        ]
    else:
        served = [(c, years, years.held) for c, years in counted]
    served = [(c, years, _in_periods(c, held)) for c, years, held in served]
    window = cast("Mapping[str, object]", identity.get("requested_window") or {"kind": "all"})
    every = frozenset[int]().union(*(years for _, _, years in served))
    elected = max(every) if window.get("kind") == "latest" and every else None
    return [ServedComponent(c, years, held, _members(c), elected) for c, years, held in served]


def _in_periods(component: ComponentIdentity, years: frozenset[int]) -> frozenset[int]:
    """`years` within `component`'s recipe periods, all of them where it declares none."""
    periods = component.recipe.periods
    return years if periods is None else frozenset(y for y in years if periods.holds(y))


def _read_years(component: ComponentIdentity, *, facts: ScopeFacts) -> ComponentYears:
    """The years `component` reads in `facts`' coverage snapshot
    (`materialise.composite.read_component_years`), read once per identity for one
    `facts`."""
    key = f"years:{component.role}:" + json.dumps(
        component.identity, sort_keys=True, separators=(",", ":")
    )
    return facts.remembered(
        key, lambda: read_component_years(facts.index, facts.coverage.index, component)
    )


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
    deadline). Each component's preview reads rows in its preview years alone
    (`component_preview`), so its series, points and counts stand as they are: each keeps
    its own StatVar, and the composite's `total_obs` sums the components'. No zero rows are
    written for a zero-filling stacked component, since the composite's coverage counts
    none. `truncated` is true when a component preview holding a series was, and `order`
    is those previews' when they agree, else None. It serves a continuation
    (`SpecObservations.continued`) when a component's read does, serves none when every
    component previewing a year read and serves none, and states nothing of one otherwise,
    when a read is lost and no other states one.

    With no series, the preview is omitted: `observations_no_points` when every component
    serving a year read and found no point in the years it serves; else the first omission
    (in component order) of another reason a component's preview carries; else, when a
    component serving a year has no read, `observations_probe_failed`. A component
    previewing no year (`ServedComponent.preview_years`) is not read, so it bears on none of
    these."""
    series: list[ObservationSeries] = []
    total = 0
    truncated = False
    orders: set[PreviewOrder | None] = set()
    for read in reads:
        if read is None or read.omitted is not None or not read.series:
            continue
        series.extend(read.series)
        total += read.total_obs
        truncated = truncated or read.truncated
        orders.add(read.order)
    previewed = [read for item, read in zip(served, reads, strict=True) if item.preview_years]
    continued = (
        True
        if any(read is not None and read.continued for read in reads)
        else False
        if all(read is not None and read.continued is False for read in previewed)
        else None
    )
    if series:
        order = next(iter(orders)) if len(orders) == 1 else None
        composed = SpecObservations(
            series=series, total_obs=total, truncated=truncated, order=order
        )
        return composed.stating_continued(continued)
    return _composite_omission(previewed).stating_continued(continued)


def component_spec_id(item: ServedComponent) -> str:
    """The spec_id of the identity `item`'s preview resolves (`component_preview`), its window
    included, with each role bound to a population open within its members
    (`identity_edit.open_within`), as `resubmit.resolve_spec_resubmit` resolves it."""
    document, _years = component_preview(item)
    return resolved_spec_id(BaseIdentity.model_validate(document), item.component.populations)


def resolved_spec_id(identity: BaseIdentity, populations: Mapping[str, RolePopulation]) -> str:
    """The spec_id of `identity` posted with `populations`, as `resubmit.resolve_spec_resubmit`
    resolves it: each role `populations` lists open within its members
    (`identity_edit.open_within`), which leaves an identity already open as it is."""
    opened, _within = open_within(identity, populations)
    return identity_spec_id(opened)


def component_preview(item: ServedComponent) -> tuple[dict[str, object], frozenset[int]]:
    """The identity `item`'s preview reads and the years it reads rows in
    (`ServedComponent.preview_years`).

    The identity is the component's own, so its read elects the facets a Datasets mint of
    the composite elects under the same window, and its window passes every gate a mint's
    read of it passes. Its read plan then reads rows over the span of the years while every
    election keeps that window (`read_plan.read_in_years`), and keeps the points in the
    years alone, so the preview's series and points lie in those years. Its counts follow
    the span of the served years and are exact when the served years are contiguous. A
    composite under a `latest` window, which no mint serves, previews the year it elects
    (`ServedComponent.elected`): each component serving that year reads it alone as a
    range, so every component plan reads a range or every date."""
    identity = dict(item.component.identity)
    if item.elected is not None:
        identity["requested_window"] = {
            "kind": "range",
            "start_year": item.elected,
            "end_year": item.elected,
        }
    return identity, item.preview_years


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


@dataclass(frozen=True, eq=False)
class CompositeMint:
    """What `spec_build.build_spec` mints a Spec as the composite `identity` from: its
    identity and `spec_id`, its components with the years each serves
    (`served_components`), the part's `facts`, and the request's `catalogue`, which its
    provenance facet and download licence read (`ComponentReads.catalogue`), else the bound
    one.

    Each method returns the value of one field the composite states in place of the table
    identity's answer (`code_index.composites.table_identity`), from that answer's own value
    where the field derives from it. Every other field, the slots and entities among them, is
    the table identity's: the composite reads the same StatVars and places."""

    identity: BaseIdentity
    spec_id: str
    served: tuple[ServedComponent, ...]
    facts: ScopeFacts
    catalogue: Catalogue | None = None
    # The identity the answer's rows were read for, the composite's table identity, posted or
    # with its population roles already open (`resolved_spec_id` hashes both alike), and the
    # roles it binds to a population (`ResubmitIdentityPayload.populations`). None where the
    # minter read no table of its own, as a same-shape merge does.
    read: BaseIdentity | None = None
    read_populations: Mapping[str, RolePopulation] = field(default_factory=dict)

    @cached_property
    def declaration(self) -> CompositeDeclaration:
        """The composite `identity`'s provenance facet binds."""
        return _declaration(
            self.facts.index, self.identity.model_dump(mode="json", exclude_none=True)
        )

    @cached_property
    def table_components(self) -> tuple[tuple[str, frozenset[int]], ...]:
        """The spec_id and preview years (`ServedComponent.preview_years`) of each component
        previewing a year whose preview resolves the read the answer made (`read`). The
        identity the component's preview resolves hashes to the read's, both with their
        population roles open (`component_spec_id`, `resolved_spec_id`), window included, and
        the component reads within the same populations (`read_populations`). Empty where the
        answer made no read of its own. A component the hash does not settle is none of
        them. Hashed once per mint, where `spec_build.build_spec` mints it, off the event
        loop."""
        if self.read is None:
            return ()
        populations = dict(self.read_populations)
        read_id = resolved_spec_id(self.read, populations)
        return tuple(
            (read_id, item.preview_years)
            for item in self.served
            if item.preview_years
            and dict(item.component.populations) == populations
            and component_spec_id(item) == read_id
        )

    @cached_property
    def years(self) -> list[int]:
        """The years some component serves, in order."""
        return sorted({year for item in self.served for year in item.served})

    @cached_property
    def window(self) -> TimeWindow | None:
        """The window the served years span, None where no component serves a year."""
        years = self.years
        return TimeWindow(start_year=years[0], end_year=years[-1]) if years else None

    def coverage(self) -> CoverageExact:
        """Every component's series counted in the years it serves, over `window`."""
        observations = sum(
            item.years.counts.get(year, 0) for item in self.served for year in item.served
        )
        return CoverageExact(
            has_data=observations > 0, observation_count=observations, window=self.window
        )

    def headline(self, stated: Headline | None) -> Headline | None:
        """The composite's headline (`_headline`), naming `stated`'s window."""
        with self._reading():
            return _headline(self.served, stated, facts=self.facts)

    def parties(self, axis: RoleAxis) -> list[str] | None:
        """The members on `axis` of the series every component writes in the years it
        serves, each read under its own places (`_served_parties`), without the totals
        `party_split` leaves out: its parties and residual members, as the frame holds
        them, sorted by ref as a table answer's scope is. These are the members of the scope
        `open_axes` states. None where the snapshot cannot list some serving component's
        parties (`_lists_parties`)."""
        split = self.party_split(axis)
        return None if split is None else sorted((*split.parties, *split.residuals))

    def party_split(self, axis: RoleAxis) -> PartySplit | None:
        """The members `parties(axis)` lists, split into parties and residual members as
        `enrichment.party_split` splits them in the components' datasets. Its counts are
        those of each `RoleAxisFact` `open_axes` states, and its party count the one the
        `role_scope_defaulted` warning states (`role_scope_warnings`). None where they
        cannot be listed."""
        held = self._held_parties
        if held is None:
            return None
        return party_split(
            held[AXIS_COVERAGE_ROLES[axis]],
            axis=axis,
            parties=self.facts.index.parties,
            datasets=self._datasets,
        )

    @cached_property
    def _datasets(self) -> list[str]:
        return [recipe.dataset for recipe in self.declaration.components]

    @cached_property
    def _held_parties(self) -> dict[Role, frozenset[str]] | None:
        coverage = self.facts.coverage.index
        with self._reading():
            unlisted = [
                item for item in self.served if item.served and not _lists_parties(item, coverage)
            ]
            if unlisted:
                logger.warning(
                    "composite %s keeps the open axes of its table read: the snapshot cannot "
                    "list the parties of %s",
                    self.declaration.provenance,
                    ", ".join(item.component.recipe.dataset for item in unlisted),
                )
                return None
            return _served_parties(self.served, coverage)

    def open_axes(
        self, scope: ResolvedScope | None, facts: list[RoleAxisFact] | None
    ) -> tuple[ResolvedScope | None, list[RoleAxisFact] | None]:
        """`scope` and `facts`, the table identity's answer's open role axes, over every
        component: the scope lists, and each counted fact counts, `parties` of its axis. A
        stacked composite's imputed line can hold recipients its bilateral line lacks, which
        the table identity's answer, reading the bilateral table alone, leaves out. A
        member the answer listed keeps its label and type, and one only another component
        holds is labelled from the code index. Both stand where the answer leaves no axis
        open, and where the snapshot cannot list the parties, which `parties` logs, since
        the answer then counts the table identity's parties alone."""
        if (scope is None and not facts) or self._held_parties is None:
            return scope, facts
        counted = facts and [
            fact.counting(split)
            if fact.member_count is not None and (split := self.party_split(fact.role)) is not None
            else fact
            for fact in facts
        ]
        if scope is None or scope.axis == "observationAbout":
            return scope, counted
        listed = {member.ref: member for member in scope.members}
        parties = self.facts.index.parties
        members = [
            listed.get(dcid)
            or ResolvedEntity(
                ref=dcid, label=row.label if (row := parties.get(dcid)) else None, role=scope.axis
            )
            for dcid in self.parties(scope.axis) or ()
        ]
        return scope.model_copy(update={"members": members}), counted

    def stat_vars(self, stated: ResolvedStatVars) -> ResolvedStatVars:
        """`stated`, the table identity's StatVar counts, over every component (`_stat_vars`)."""
        with self._reading():
            return _stat_vars(self.served, stated, self.declaration, facts=self.facts)

    def slots(self, stated: Sequence[Slot]) -> list[Slot]:
        """`stated` with the when slot bound to `window` (`_window_binding`), where some
        component serves a year."""
        window = self.window
        return [
            slot.model_copy(update={"binding": _window_binding(window)})
            if slot.key.axis == "when" and window is not None
            else slot
            for slot in stated
        ]

    def download(self) -> DownloadFacts | None:
        """The facts of the composite file a Datasets mint writes (`composite_download`)."""
        with self._reading():
            return composite_download(self.served, self.identity, facts=self.facts)

    def measure_label(self, stated: str | None) -> str | None:
        """The measure label the composite's sentence names (`_measure_label`)."""
        return _measure_label(self.declaration, stated)

    def sources(self) -> list[ProvenanceRef]:
        """The component tables, each under its declared label (`_component_sources`)."""
        return _component_sources(self.facts.index, self.declaration)

    def published_tables(self) -> str | None:
        """The component tables' labels, joined."""
        return join_labels([source.label for source in self.sources()])

    def provenance_facet(self) -> ResolvedFacet:
        """The provenance facet naming the composite, with its label, the source and Data
        Commons its component tables share (`search.observations.composite_facet_ref`), and
        the account they give."""
        with self._reading():
            ref = composite_facet_ref(self.declaration, index=self.facts.index)
        return ResolvedFacet(
            basis="bound",
            ref=ref.ref,
            label=ref.label,
            source=ref.source,
            data_commons=ref.data_commons,
            account=provenance_account(self.facts.index, self.declaration.provenance),
        )

    def facets(self, stated: Mapping[str, ResolvedFacet]) -> dict[str, ResolvedFacet]:
        """`stated` with the provenance facet naming the composite."""
        return {**stated, "provenance": self.provenance_facet()}

    def data_years(self) -> tuple[int, int] | None:
        """The first and last year some component serves (`ResolvedSpec.data_years`)."""
        years = self.years
        return (years[0], years[-1]) if years else None

    def components(self) -> list[Component]:
        """The composite's components (`CompiledScope.components`)."""
        return _components(self.served)

    def role_scope_warnings(self, warnings: Sequence[Warning], spec: Spec) -> tuple[Warning, ...]:
        """`warnings`, with each `role_scope_defaulted` warning about `spec`, the Spec minted
        from this composite, worded to count the `parties` of its `params["axis"]`, as a
        fresh answer's warning states its count (`grounding.role_scope_defaulted_message`).
        The count is the one `spec`'s role axis fact states (`open_axes`), so the warning
        and the Spec agree where the snapshot cannot list the parties and the fact keeps the
        table's count. A warning is about `spec` where it is keyed to `spec` or to no Spec.
        One keyed to another reading, such as a flagged match or a suggestion beside the
        answer, counts that reading's parties and keeps its words. An axis on which `spec`
        has no counted role axis fact keeps its warning, as does every other warning."""
        # Deferred: grounding imports this module while it loads.
        from search.engine.region.codes import ROLE_SCOPE_DEFAULTED
        from search.engine.region.grounding import role_scope_defaulted_message

        counted = {
            fact.role: (fact.member_count, fact.residual_count or 0)
            for fact in reversed(spec.resolved.role_axis_facts or ())
            if fact.member_count is not None
        }
        labels = {entity.ref: entity.label or entity.ref for entity in spec.resolved.entities}

        def stated(warning: Warning) -> Warning:
            axis = (warning.params or {}).get("axis")
            if (
                warning.code != ROLE_SCOPE_DEFAULTED
                or warning.spec_id not in (None, spec.spec_id)
                or axis not in counted
            ):
                return warning
            role = cast("RoleAxis", axis)
            opposite: EntityRoleKey = "origin" if role == "destination" else "destination"
            member_count, residual_count = counted[role]
            message = role_scope_defaulted_message(
                role,
                member_count=member_count,
                residual_count=residual_count,
                bound_dcids=list(dict.fromkeys(self.identity.entities.get(opposite, ()))),
                label_of=lambda dcids: [labels.get(dcid, dcid) for dcid in dcids],
            )
            return warning.model_copy(update={"message": message})

        return tuple(map(stated, warnings))

    @contextmanager
    def _reading(self) -> Iterator[None]:
        """Binds `catalogue`, where one is given, for the body of the `with` block."""
        if self.catalogue is None:
            yield
            return
        with bound_catalogue(self.catalogue):
            yield


def _lists_parties(item: ServedComponent, coverage: CoverageIndex) -> bool:
    """Whether the snapshot lists the parties of the series `item` writes: it settles its
    members and holds a row of each (`CoverageIndex.covers`), as `composite_download`
    counts rows only there."""
    return item.members is not None and all(coverage.covers(sv) for sv in item.members)


def _served_parties(
    served: Sequence[ServedComponent], coverage: CoverageIndex
) -> dict[Role, frozenset[str]]:
    """The donors and recipients of the series each of `served` writes in the years it
    serves (`_served_series`), as the mint's frame holds them, for components whose parties
    the snapshot lists (`_lists_parties`)."""
    parties: dict[Role, set[str]] = {"recipient": set(), "donor": set()}
    for item in served:
        if not item.served:
            continue
        for (_variable, recipient, donor), _held in _served_series(item, coverage):
            parties["recipient"].add(recipient)
            parties["donor"].add(donor)
    return {role: frozenset(held - {ABSENT_ENTITY}) for role, held in parties.items()}


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
    provenance refs a sentence's byline and a citation name (`observations.provenance_refs`),
    each under its declared label, as a plain answer names its table
    (`spec_build.elected_sources`)."""
    datasets = [recipe.dataset for recipe in declaration.components]
    labels = {dataset: index.datasets[dataset].label for dataset in datasets}
    return [
        ref.model_copy(update={"label": labels[ref.ref]})
        for ref in provenance_refs(datasets, labels)
    ]


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
    """The StatVars of `item` holding a series under `bound` in one of `years`, read under
    the provenances its identity pins alone (`materialise.composite.component_reads`)."""
    held = sorted(set(years))
    if item.members is None or not held:
        return frozenset()
    with component_reads(item.component.identity):
        found = coverage.series_years(
            variables=sorted(item.members),
            bound=bound,
            first_year=held[0],
            last_year=held[-1],
            include=window_include(item.component.identity),
        )
    if found is None:
        return frozenset()
    series = found.keys()
    return frozenset(variable for variable, _, _ in series)


def _stat_vars(
    served: Sequence[ServedComponent],
    stated: ResolvedStatVars,
    declaration: CompositeDeclaration,
    *,
    facts: ScopeFacts,
) -> ResolvedStatVars:
    """`stated`, the table identity's StatVar counts, counted over every component of the
    composite `declaration`. The member count is the StatVars holding a series under their
    component's own places (`_bound`) in the years it serves, the served count sums the
    StatVars a mint of each component serves (None where the index does not settle some
    component's members), the served date
    range spans the years the components serve, and `sums_components` states whether a
    groupby sum adds the components (`code_index.composites.components_sum`), the rule the
    mint sums by, and `splices_components` whether it splices them
    (`code_index.composites.components_splice`), the rule the mint splices by."""
    years = sorted({year for item in served for year in item.served})
    confirmed = sum(
        len(
            _with_data(
                item, bound_roles(item.component.identity), facts.coverage.index, item.served
            )
        )
        for item in served
    )
    counts = [len(item.members) if item.members is not None else None for item in served]
    return stated.model_copy(
        update={
            "member_count": confirmed,
            "member_count_exact": True,
            "served_count": None if None in counts else sum(c for c in counts if c is not None),
            "sums_components": components_sum(declaration),
            "splices_components": components_splice(declaration),
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

    Each component writes its members' series-years in the years it serves
    (`ServedComponent.served`, within its recipe's periods). A zero-filling component
    (`ComponentRecipe.zero_absent`) writes, for each member a mint of it serves, one row per
    cell, a (recipient, donor, year), that some component's series fills in the years it
    serves, its own rows and its zero rows alike. Series count the distinct (StatVar,
    recipient, donor) keys written, zero-filled ones included. The window spans the years
    written. The licence is the one every component table's source-index entry names, else
    None.

    The count reads no facet: it keeps a member's rows under a source, unit or method the
    mint's facet election leaves out, and counts the zero rows of such a member, and under a
    unit request it keeps rows a conversion would drop (`unit.conversion.counted_conversion`).
    So `rows` can overstate the file, and `rows_exact` is always false. A component the
    snapshot settles as serving no year (`YearCounts.settled`) writes no row, so it counts
    nothing whatever its members. `rows`, `series` and `window` are None where some other
    component has members not settled in memory
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
    writing = [item for item in served if item.served or not item.years.settled]
    if any(
        item.members is None
        or not (
            item.component.recipe.zero_absent or all(coverage.covers(sv) for sv in item.members)
        )
        for item in writing
    ):
        return DownloadFacts(rows=None, rows_exact=False, series=None, window=None, licence=licence)
    rows = series = 0
    years: set[int] = set()
    cells: set[tuple[str, str, int]] = set()
    for item in writing:
        for (_variable, recipient, donor), held in _served_series(item, coverage):
            cells.update((recipient, donor, year) for year in held)
            if not item.component.recipe.zero_absent:
                rows += len(held)
                series += 1
                years.update(held)
    places = {(recipient, donor) for recipient, donor, _year in cells}
    for item in writing:
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
    """Each series of `item`'s members under its identity's places, read under the
    provenances its identity pins alone (`materialise.composite.component_reads`), with the
    years of it that `item` serves, for a series holding one."""
    if item.members is None or not item.served:
        return
    with component_reads(item.component.identity):
        found = coverage.series_years(
            variables=sorted(item.members),
            bound=bound_roles(item.component.identity),
            first_year=min(item.served),
            last_year=max(item.served),
            include=window_include(item.component.identity),
        )
    if found is None:
        return
    epoch = coverage.horizon.epoch_year
    for key, bits in zip(found.keys(), found.year_bits(), strict=True):
        held = frozenset(
            year
            for offset in range(bits.bit_length())
            if bits >> offset & 1 and (year := epoch + offset) in item.served
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
    stated: Headline | None,
    *,
    facts: ScopeFacts,
) -> Headline | None:
    """The composite's headline: the latest year any component serves, counted over the
    rows of every component serving it, each under its own places and the provenances it
    pins (`_headline_year`), with the same counts for it and the two years before it among
    those the composite serves. A
    component reads its own places, so one that binds its destinations to a population
    (`ComponentIdentity.populations`) counts that population alone. The named window is
    `stated`'s, the request's. None when no component serves a year or settles its
    members."""
    coverage = facts.coverage.index
    years = sorted({year for item in served for year in item.served}, reverse=True)[:3]
    counted: list[HeadlineYear] = []
    for year in years:
        reads: dict[_ReadKey, set[str]] = {}
        for item in served:
            if year in item.served and item.members is not None:
                identity = item.component.identity
                key = (
                    tuple(sorted(bound_roles(identity).items())),
                    _pins(identity, facts.index),
                    window_include(identity),
                )
                reads.setdefault(key, set()).update(item.members)
        found = _headline_year(coverage, reads, year)
        if found is not None:
            counted.append(found)
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


def _pins(identity: Mapping[str, object], index: CodeIndex) -> frozenset[str] | None:
    """The provenances `identity` pins: those a read naming its pinned refs admits under
    `index` (`code_index.members.pinned_refs`, `source_scope.read_provenances`), every member
    provenance of a declared dataset. None for none."""
    facets = cast("Mapping[str, Mapping[str, object]]", identity.get("facets") or {})
    return read_provenances(pinned_refs(facets), code_index=index) or None


type _ReadKey = tuple[
    tuple[tuple[Role, frozenset[str]], ...], frozenset[str] | None, frozenset[str]
]
"""The places a component binds, as sorted items of `bound_roles`, the provenances it pins
(`_pins`) and the held-back years its window opts in to (`window_include`). The components
of one key read their series together."""


def _headline_year(
    coverage: CoverageIndex, reads: Mapping[_ReadKey, set[str]], year: int
) -> HeadlineYear | None:
    """The counts of `year` over `reads`, the StatVars of the components serving it grouped
    by the places they bind and the provenances they pin (`_ReadKey`), each group read under
    its pins alone (`origin_catalogue.pinned_reads`). That is `CoverageIndex.headline` of one
    group, or across several the summed observations and series of each group's, whose
    StatVars differ, and the distinct parties on each role across all of their series in the
    year. None when no row holds `year`."""
    found: list[
        tuple[CoverageHeadline, BoundRoles, frozenset[str] | None, frozenset[str], list[str]]
    ] = []
    for (places, pins, include), members in reads.items():
        variables = sorted(members)
        bound: BoundRoles = dict(places)
        with pinned_reads(pins):
            held = coverage.headline(
                variables=variables,
                bound=bound,
                first_year=year,
                last_year=year,
                include=include,
            )
        if held is not None and held.year == year:
            found.append((held, bound, pins, include, variables))
    if not found:
        return None
    names = coverage.role_names.get(found[0][4][0])
    role_dcids = {} if names is None else {"donor": names.donor, "recipient": names.recipient}
    if len(found) == 1:
        role_counts = found[0][0].role_counts
    else:
        role_counts = _distinct_parties(coverage, [(b, p, i, v) for _, b, p, i, v in found], year)
    return HeadlineYear(
        year=year,
        observations=sum(held.observations for held, *_ in found),
        series=sum(held.series for held, *_ in found),
        role_counts={
            dcid: count
            for role, count in role_counts.items()
            if (dcid := role_dcids.get(role)) is not None
        },
    )


def _distinct_parties(
    coverage: CoverageIndex,
    reads: Sequence[tuple[BoundRoles, frozenset[str] | None, frozenset[str], list[str]]],
    year: int,
) -> dict[Role, int]:
    """The distinct donors and recipients of the series `reads` hold in `year`, each read
    under its own bound, pins (`origin_catalogue.pinned_reads`) and opt-in, by coverage
    role; a role no series carries a party on is left out."""
    parties: dict[Role, set[str]] = {"recipient": set(), "donor": set()}
    for bound, pins, include, variables in reads:
        with pinned_reads(pins):
            found = coverage.series_years(
                variables=variables,
                bound=bound,
                first_year=year,
                last_year=year,
                include=include,
            )
        if found is None:
            continue
        series = found.keys()
        for _variable, recipient, donor in series:
            parties["recipient"].add(recipient)
            parties["donor"].add(donor)
    counts = {role: len(held - {ABSENT_ENTITY}) for role, held in parties.items()}
    return {role: count for role, count in counts.items() if count}
