"""The provider axis: whose spending a part counts, the donor's bilateral flows alone, with
its imputed share of multilateral spending, with its core contributions to multilaterals, or
the core contributions alone.

The inclusive scope binds a stacked composite of the part's bilateral flows and an imputed
multilateral line as the identity's source; on the CRS the bilateral scope removes it. Each
table that offers the scope declares its composite and the measure bases it adds on
(`code_index.scope_conventions.INCLUSIVE_SOURCES`): the CRS with ONE's imputed multilateral
table on gross disbursements, and DAC2a's total ODA with its own imputed multilateral ODA on
gross and net flows. The axis is present on a part the CRS serves and on a donor's published
total (`provider_rule`). A published total holds the scope its declared row names
(`code_index.scope_conventions.PUBLISHED_PROVIDER_SCOPES`): DAC1's total ODA the contributions
scope, its bilateral ODA the bilateral scope and its multilateral ODA the multilateral scope,
and DAC2a's total ODA the contributions scope, since it lists the multilateral agencies among
its recipients and counts the donor's core contributions to them. Each DAC1 total offers the
others' scopes as options binding their aid types. Such an option serves the years every donor
holding its own total in a year also holds the other in, or, for a part asking for the latest
year, the one year the pool's read elects (`_published_table`). DAC2a's bilateral scope binds a
composite of its total ODA that reads the bilateral destinations alone
(`code_index.scope_conventions.BILATERAL_SOURCES`): countries, regions and the DAC's
unallocated lines. One rule reads the destinations a part names (`destination_rule`): on a part
naming a multilateral agency the bilateral and inclusive scopes are unavailable
(`multilateral_destination`, naming the agency), and on a part naming all recipients they read
the DAC's total of developing countries in its place, as their labels say (`_rebound_label`).
Over destinations on the bilateral side alone the total holds no core contribution, so a
composite reading offers no switch to it. Where a table declares an inclusive source, the
inclusive option serves the years both lines cover for the part (`inclusive_reach`): available
when they cover every year the bilateral answer holds, partial with a shorter window of its own
when they cover fewer, and, for a part asking for the latest year, available over the year a
`latest` read elects among those both cover. It is not offered (`provider_not_offered`) where
the imputed line cannot serve what the part binds, or cannot count the providers it names
(`_providers_not_offered`): a part naming no provider, a provider the line carries no rows for,
or a pool naming a multilateral whose outflows the line attributes to its funders. On any other
basis the source does not add on it conflicts with the measure-basis option
(`AxisTable.conflicts`).

The default is declared (`code_index.scope_conventions.PROVIDER_CONVENTIONS`): a named donor's
total of a CRS sector or recipient part counts both, when they cover every year asked for.
`settle_provider` applies it, or the part's explicit or carried entry, on the parsed door,
where the answer can serve it (`_served_in_pass`). A fresh DAC2a part counting total ODA
reads its bilateral destinations before the axis is read (`bilateral_default_reading`), where
it names no destination or only bilateral ones, whatever donors it names. A choice names the
convention's rule only where the request matched the convention and the part holds its scope.
Otherwise it names the rule of the option it holds, its own scope's (`AxisTable.rules`), as
every resubmit does.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal, cast, get_args

from code_index.composites import (
    CompositeNotServable,
    PopulationUnavailable,
    component_identities,
    composite_ref,
    population_check,
    table_identity,
    without_composite,
)
from code_index.members import SourceNotInFamily, identity_family_query, index_family
from code_index.query import CodeIndex
from code_index.scope_conventions import (
    BASIS_MEASURES,
    BILATERAL_SOURCES,
    DONOR_ORGANISATIONS,
    INCLUSIVE_SOURCES,
    OPTION_LABELS,
    PROVIDER_CONVENTIONS,
    PROVIDER_SCOPE_RULES,
    PUBLISHED_PROVIDER_SCOPES,
    REDISTRIBUTED_PROVIDERS,
    BilateralSource,
    InclusiveSource,
    ProviderConvention,
    PublishedProviderScope,
    destination_side,
)
from dfo_platform.roles import role_key_axis
from materialise.composite import spans
from search.catalogue.counts import COUNTRY_PREFIX
from search.engine.spec_id import identity_spec_id
from search.models import (
    BaseIdentity,
    ChoiceOrigin,
    Entry,
    IdentityBindingSet,
    IdentityBindingValue,
    IdentityFacetValueRequest,
    IdentityShapeSchema,
    IdentityWindow,
    IdentityWindowLatest,
    IdentityWindowRange,
    Option,
    OptionAvailability,
    OptionBinding,
    ProviderBinding,
    ProviderScope,
    ScopeReasonCode,
    SlotBinding,
    TextOrigin,
    YearSpan,
)
from search.scope.basis import held_measure, identity_part_shape
from search.scope.composite import (
    ServedComponent,
    component_years,
    donor_years,
    served_components,
)
from search.scope.edits import (
    applies_in_pass,
    apply_binding,
    counted_bilateral_source,
    rebound_destinations,
)
from search.scope.region import policy_members, policy_members_between
from search.scope.state import AxisRecord, ScopeState
from search.scope.tables import AxisTable, Conflict, ScopeFacts
from search.scope.vocabulary import measure_basis_option_id, provider_option_id
from series_coverage.index import elected_year

BILATERAL = provider_option_id("bilateral")
"""The bilateral provider option."""
INCLUSIVE = provider_option_id("inclusive")
"""The inclusive provider option."""
CONTRIBUTIONS = provider_option_id("contributions")
"""The contributions provider option: bilateral flows plus core contributions to
multilaterals, DAC1's total ODA."""
MULTILATERAL = provider_option_id("multilateral")
"""The multilateral provider option: core contributions to multilaterals alone, DAC1's
multilateral ODA."""
_SCOPE_LABELS: Mapping[ProviderScope, str] = {
    "bilateral": OPTION_LABELS.provider_bilateral,
    "inclusive": OPTION_LABELS.provider_inclusive,
    "contributions": OPTION_LABELS.provider_contributions,
    "multilateral": OPTION_LABELS.provider_multilateral,
}

_SOURCES: Mapping[str, InclusiveSource] = {row.dataset: row for row in INCLUSIVE_SOURCES}
"""The inclusive source of each table that declares one, by the table's provenance."""
_SOURCE_COMPOSITES = frozenset(row.composite for row in INCLUSIVE_SOURCES)
_BILATERAL_SOURCES: Mapping[tuple[str, str, str], BilateralSource] = {
    (row.dataset, row.code_system, row.code): row for row in BILATERAL_SOURCES
}
"""The bilateral source of each published total that declares one, by the total's dataset,
code system and code."""
_PROVIDER_COMPOSITES = _SOURCE_COMPOSITES | frozenset(row.composite for row in BILATERAL_SOURCES)
"""Every composite a provider option binds as the identity's source."""
_PUBLISHED_DATASETS = frozenset(row.dataset for row in PUBLISHED_PROVIDER_SCOPES)
"""The tables whose provider axis their published rows declare, on those totals alone."""
_COUNTS_CONTRIBUTIONS: frozenset[ProviderScope] = frozenset({"contributions", "multilateral"})
"""The scopes that count a donor's core contributions to multilaterals as it pays them, which
an imputed share of the multilaterals' spending would count a second time."""
_INCLUSIVE_BASIS_CAUSE = "imputed_gross_disbursements_only"
_REDISTRIBUTED = frozenset(row.provider for row in REDISTRIBUTED_PROVIDERS)
_SCOPE_RULES: Mapping[ProviderScope, str] = {row.scope: row.rule for row in PROVIDER_SCOPE_RULES}
"""The rule each provider scope names where no convention or published row settles it."""
_DONOR_ORGANISATIONS = frozenset(DONOR_ORGANISATIONS)
_NET = "net"
_CONSTANT = "RealValue"
logger = logging.getLogger(__name__)
_SLOT_NOT_SERVED = "slot_not_served"
"""The `why` of `provider_not_offered` where a provider composite cannot serve a slot the part
binds, whose property the detail's `slot` names (`_composite_fault`)."""
_COMPOSITE_NOT_SERVED = "composite_not_servable"
"""The `why` of `provider_not_offered` where a provider composite refuses what the part binds
at a member that names no slot (`/entities/<key>`, `/facets/provenance`, `/shape`), the
member in the detail's `pointer` (`_composite_fault`)."""

WithheldWhy = Literal[
    "destination_unclassified",
    "option_not_covered",
    "read_window_not_covered",
    "scope_undecided",
    "slot_not_served",
]
"""Why a fresh reading the bilateral default reaches on a part with no provider axis reads total
ODA: the reasons dfo-spec-kit's `registries/warning-reasons.json` lists for
`bilateral_reading_unavailable`, and no others."""
WITHHELD_WHYS: frozenset[str] = frozenset(get_args(WithheldWhy))


@dataclass(frozen=True, slots=True)
class Withheld:
    """Why the bilateral default reaches a reading and cannot apply (`WithheldWhy`), and for
    `slot_not_served` the property of the slot the bilateral composite cannot serve."""

    why: WithheldWhy
    slot: str | None = None

    @property
    def params(self) -> dict[str, str]:
        """The `bilateral_reading_unavailable` warning's `params`: `why`, and `slot` where the
        reason names one."""
        return {"why": self.why} if self.slot is None else {"why": self.why, "slot": self.slot}


@dataclass(frozen=True, slots=True)
class InclusiveReach:
    """What the inclusive scope serves of a part: `asked`, the years its bilateral answer
    holds in the window it asks for, and `covered`, those of them the composite serves, in
    the years both tables cover (`composite.served_components`). For a part asking for the
    latest year, `latest` is the year a `latest` read elects
    (`series_coverage.index.elected_year`) over `covered`, counting the series of both
    tables in each; None otherwise, or when `covered` is empty."""

    asked: frozenset[int]
    covered: frozenset[int]
    latest: int | None = None


def inclusive_reach(identity: BaseIdentity, facts: ScopeFacts, *, composite: str) -> InclusiveReach:
    """The years the inclusive scope binding `composite` serves of a part whose bilateral
    identity is `identity` (`InclusiveReach`), read from the coverage snapshot in memory: the
    years every component serves (`composite.served_components`), so a line that reads a
    missing row as unknown leaves out the years it holds none in. Raises
    `code_index.composites.CompositeNotServable` where the imputed line cannot serve what
    `identity` binds."""
    document = identity.model_dump(mode="json", exclude_none=True)
    served = served_components(_with_composite(document, composite), facts=facts)
    bilateral = next(c for c in served if c.component.role == "bilateral")
    covered = frozenset[int].intersection(*(c.served for c in served))
    if document["requested_window"]["kind"] == "latest":
        return InclusiveReach(
            asked=bilateral.years.held, covered=covered, latest=elected_year(_served_counts(served))
        )
    return InclusiveReach(asked=bilateral.years.held, covered=covered)


def _served_counts(served: Sequence[ServedComponent]) -> dict[int, int]:
    """The series per year a composite serves, summed over its `served` components
    (`composite.served_components`), in the years every component serves, so a `latest` read
    of the composite and an option electing its year count alike."""
    if not served:
        return {}
    covered = frozenset[int].intersection(*(c.served for c in served))
    return {year: sum(c.years.counts.get(year, 0) for c in served) for year in covered}


def _with_composite(document: Mapping[str, object], composite: str) -> dict[str, object]:
    """`document` binding `composite` as its source."""
    facets = cast("Mapping[str, object]", document.get("facets") or {})
    return {**document, "facets": {**facets, "provenance": {"state": "value", "ref": composite}}}


def provider_rule(
    base: BaseIdentity, record: AxisRecord | None, facts: ScopeFacts
) -> AxisTable | None:
    """The provider table of a part whose identity is `base`, None where the axis does not
    apply to it: `base` counts no published total (`_published_row`), and binds a composite
    other than its table's inclusive source or, without it, is served from a table that
    declares no inclusive source (`INCLUSIVE_SOURCES`) or one whose published rows declare its
    provider axis (`PUBLISHED_PROVIDER_SCOPES`).

    Read from `base`, so a resubmit of a served identity offers what its search offered. A
    published total offers what `_published_table` states. On a part the CRS serves, the
    option `base` holds leads the preference, so a part's default is the scope it resolved;
    the bilateral option is always available, and the inclusive option is available, partial
    or unavailable by `inclusive_reach`, or not offered (`_not_offered`). Each option names
    its own rule (`AxisTable.rules`): its row's on a published total, and its scope's
    (`PROVIDER_SCOPE_RULES`) on the CRS or where no row declares the scope. A fresh search's
    `record` names the rule that settled its option (`settle_provider`), which the table
    names for that option, so a resubmit, which settles no convention, names the option's
    own rule. The table of `base` is read once per scope build."""
    key = "provider_table:" + base.model_dump_json(exclude_none=True)
    table = facts.remembered(key, lambda: _table(base, facts))
    if table is None or record is None or table.option(record.active) is None:
        return table
    return dataclasses.replace(table, rules={**table.rules, record.active: record.rule})


def _table(base: BaseIdentity, facts: ScopeFacts) -> AxisTable | None:
    """The provider table of `base` (`provider_rule`), None where the axis does not apply."""
    published = _published_row(base, facts)
    if published is not None:
        return _published_table(published, base, facts=facts)
    try:
        ref = composite_ref(base.model_dump(mode="json", include={"facets"}))
    except CompositeNotServable:
        return None
    if ref is not None and ref not in _SOURCE_COMPOSITES:
        return None
    bilateral = _without_provider_composite(base, ref)
    dataset = _dataset(bilateral, facts.index)
    source = _SOURCES.get(dataset) if dataset not in _PUBLISHED_DATASETS else None
    if source is None or ref not in (None, source.composite):
        return None
    held = BILATERAL if ref is None else INCLUSIVE
    options = (
        _option("bilateral"),
        _inclusive_option(bilateral, source, facts, held=held == INCLUSIVE),
    )
    rules = {BILATERAL: _SCOPE_RULES["bilateral"], INCLUSIVE: _SCOPE_RULES["inclusive"]}
    return AxisTable(
        axis="provider",
        options=options,
        preference=(held, BILATERAL if held == INCLUSIVE else INCLUSIVE),
        held=held,
        rules=rules,
        conflicts=_basis_conflicts(source),
    )


def _basis_conflicts(source: InclusiveSource) -> tuple[Conflict, ...]:
    """The inclusive option's conflict with each measure-basis option whose basis `source`
    does not add its imputed line on (`InclusiveSource.bases`)."""
    return tuple(
        Conflict(INCLUSIVE, measure_basis_option_id(row.basis), _INCLUSIVE_BASIS_CAUSE)
        for row in BASIS_MEASURES
        if row.basis not in source.bases
    )


def _without_provider_composite(identity: BaseIdentity, ref: str | None) -> BaseIdentity:
    """`identity`, whose composite is `ref`, without a composite a provider option binds (an
    inclusive source's, or a bilateral source's on a published total): the identity of the
    table it reads. `identity` itself when `ref` is another one."""
    if ref not in _PROVIDER_COMPOSITES:
        return identity
    return BaseIdentity.model_validate(
        without_composite(identity.model_dump(mode="json", exclude_none=True))
    )


def _published_table(
    row: PublishedProviderScope, identity: BaseIdentity, *, facts: ScopeFacts
) -> AxisTable:
    """The provider table of a part counting the published total `row` declares, whose
    identity is `identity` (`_published_row`).

    The part holds `row`'s scope, the inclusive one where `identity` binds its dataset's
    inclusive source (`INCLUSIVE_SOURCES`), or the bilateral one where it binds the bilateral
    source of `row`'s total (`BILATERAL_SOURCES`). Every other row of its dataset and code
    system (`PUBLISHED_PROVIDER_SCOPES`) is an option binding that row's code, so a switch to
    it reads the total that counts its scope: on DAC1's total ODA, bilateral flows bind
    bilateral ODA and core contributions alone multilateral ODA, and on bilateral ODA, core
    contributions bind total ODA. Such an option serves the years every donor holding the
    part's total in a year also holds the sibling in (`_sibling_option`), or, on a part asking
    for the latest year, the one year the pool's read elects (`_latest_sibling_option`).
    Where `row`'s total declares a bilateral source, the bilateral option binds its composite
    (`_bilateral_option`), and the destinations the part names are read by one rule
    (`destination_rule`). Where the dataset declares an inclusive source, the inclusive option
    is offered as on a part the CRS serves (`_inclusive_option`), with its basis conflicts,
    and under the same rule (`_published_inclusive`). Otherwise it is unavailable with the
    total's dataset: `contributions_counted` on a total counting core contributions, since it
    counts them in place of an imputed share of the multilaterals' spending, and
    `provider_not_offered` (why `dataset`) on any other. Each option names its row's rule,
    the bilateral source's its rule, and the inclusive one its scope's. A total that declares
    a bilateral source names its own option by `BilateralSource.total_label`, and on a part
    holding a composite over destinations on the bilateral side alone that option is not
    offered (`provider_not_offered`, why `destination_bilateral`): those destinations hold no
    core contribution, so it reads the composite's figures. No preference but the held scope
    leads, so no default moves the part to another total: a fresh part reads the bilateral
    source before its table is read (`bilateral_default_reading`)."""
    rows = {
        sibling.scope: sibling
        for sibling in PUBLISHED_PROVIDER_SCOPES
        if (sibling.dataset, sibling.code_system) == (row.dataset, row.code_system)
    }
    source = _SOURCES.get(row.dataset)
    split = _BILATERAL_SOURCES.get((row.dataset, row.code_system, row.code))
    ref = composite_ref(identity.model_dump(mode="json", include={"facets"}))
    inclusive_held = source is not None and ref == source.composite
    bilateral_held = split is not None and ref == split.composite
    bilateral = _without_provider_composite(identity, ref)
    rule = destination_rule(bilateral, split, facts) if split is not None else None
    # Over bilateral-side destinations alone the total holds no core contribution, so a
    # switch to it from a composite reads the same figures under another label.
    total_alike = (
        rule is not None and rule.named and rule.bilateral and (bilateral_held or inclusive_held)
    )
    options: list[Option] = []
    rules: dict[str, str] = {}
    own = _donor_years(bilateral, facts)
    own_year = (
        elected_year(_year_counts(bilateral, facts))
        if bilateral.requested_window.kind == "latest"
        else None
    )
    for scope in get_args(ProviderScope):
        sibling = rows.get(scope)
        option_id = provider_option_id(scope)
        if scope == "bilateral" and rule is not None:
            rules[option_id] = rule.source.rule
            options.append(_bilateral_option(bilateral, rule, facts))
            continue
        if sibling is None and scope != "inclusive":
            continue
        rules[option_id] = sibling.rule if sibling is not None else _SCOPE_RULES[scope]
        if sibling is None:
            options.append(
                _published_inclusive(row, source, bilateral, facts, held=inclusive_held, rule=rule)
            )
            continue
        if sibling is row:
            label = split.total_label if split is not None else None
            availability = (
                _unavailable("provider_not_offered", {"why": "destination_bilateral"})
                if total_alike
                else None
            )
            options.append(
                _option(scope, availability, binding=ProviderBinding(scope=scope), label=label)
            )
            continue
        binding = SlotBinding(slot=sibling.code_system, state="value", refs=[sibling.code])
        switched = apply_binding(bilateral, binding, [])
        sibling_years = _donor_years(switched, facts)
        options.append(
            _sibling_option(scope, binding, own, sibling_years)
            if bilateral.requested_window.kind != "latest"
            else _latest_sibling_option(
                scope, binding, own, sibling_years, _year_counts(switched, facts), own_year
            )
        )
    held = (
        INCLUSIVE
        if inclusive_held
        else BILATERAL
        if bilateral_held
        else provider_option_id(row.scope)
    )
    return AxisTable(
        axis="provider",
        options=tuple(options),
        preference=(held,),
        held=held,
        rules=rules,
        published=True,
        conflicts=_basis_conflicts(source) if source is not None else (),
    )


@dataclass(frozen=True, slots=True)
class DestinationRule:
    """How the composite of `source` (`BilateralSource`), which reads the bilateral
    destinations of `source`'s total alone, reads the destinations a part names, each placed
    by `code_index.scope_conventions.destination_side` in `source`'s dataset: `multilateral`,
    the first one on the multilateral side, None when none is, where the composite holds no
    row and the part reads the total itself; `spanning`, `source`'s total
    of both sides, None when the part does not name it, which the composite reads at the
    total of the bilateral side in its place (`read`); `bilateral`, whether
    the part names no destination or only ones on the bilateral side; and `named`, whether it
    names any. A destination the index holds no row for, or that it cannot place, is on
    neither side.

    One rule for a fresh part's provider options and its default (`_published_table`,
    `with_bilateral_default`), read once per scope build (`destination_rule`). A posted
    identity that binds the composite to a destination off its side is refused at every
    Search door that reads a posted identity's table, the resubmit, candidate and options
    doors (`identity_edit.composite_table_identity`)."""

    source: BilateralSource
    multilateral: str | None
    spanning: str | None
    bilateral: bool
    named: bool

    def read(self, identity: BaseIdentity) -> BaseIdentity:
        """`identity` naming the total of the bilateral side in place of the total of both
        sides where the part names it (`edits.rebound_destinations`), `identity` itself
        otherwise."""
        if self.spanning is None:
            return identity
        return rebound_destinations(identity, self.source.composite)


def destination_rule(
    identity: BaseIdentity, source: BilateralSource, facts: ScopeFacts
) -> DestinationRule:
    """The `DestinationRule` of the destinations `identity` names against `source`, read in
    `facts`' code index once per scope build for `identity` and `source`."""
    key = f"destination_rule:{source.composite}:" + identity.model_dump_json(exclude_none=True)
    return facts.remembered(key, lambda: _destination_rule(identity, source, facts.index))


def _destination_rule(
    identity: BaseIdentity, source: BilateralSource, index: CodeIndex
) -> DestinationRule:
    named = [
        dcid
        for key, dcids in identity.entities.items()
        if role_key_axis(key) == "destination"
        for dcid in dcids
    ]
    sides = {
        dcid: destination_side(party, datasets=(source.dataset,))
        if (party := index.parties.get(dcid)) is not None
        else None
        for dcid in named
    }
    return DestinationRule(
        source=source,
        multilateral=next((d for d, side in sides.items() if side == "multilateral"), None),
        spanning=source.spanning if source.spanning in sides else None,
        bilateral=all(side == "bilateral" for side in sides.values()),
        named=bool(sides),
    )


def _population_unavailable(composite: str, facts: ScopeFacts) -> bool:
    """Whether a component of `composite` reads a population of destinations that is
    unavailable in `facts`' indexes (`code_index.composites.population_check`): a destination
    the code index or the coverage snapshot holds that no side declaration places, which the
    mint refuses to read."""
    return population_check(facts.index, facts.coverage.index).refusal(composite) is not None


def _bilateral_option(identity: BaseIdentity, rule: DestinationRule, facts: ScopeFacts) -> Option:
    """The bilateral option binding the composite of `rule`'s source on a part whose identity
    without a provider composite is `identity`, labelled by `BilateralSource.label`, read
    once per scope build for `identity` and the source. It is unavailable
    (`multilateral_destination`, naming the destination) where the part names a multilateral
    destination, whose flows are core contributions the composite leaves out, and not
    offered (`provider_not_offered`) where the composite's population is unavailable (why
    `destination_unclassified`, `_population_unavailable`) or the composite refuses what
    `identity` binds (`code_index.composites.CompositeNotServable`, `_composite_fault`: why
    `slot_not_served` naming the slot at fault, or `option_not_covered` where the window reaches
    none of the composite's years). Otherwise it serves the years its one component serves
    (`_composite_years`), judged as a sibling total is (`_sibling_option`,
    `_latest_sibling_option`, each `option_not_covered`): available when they cover every
    year the part's answer holds, partial over their latest run when they cover fewer, and
    unavailable when they cover none, as for a donor whose total ODA holds no breakdown by
    recipient. Where the part names the total of both sides, the composite reads the total
    of the bilateral side in its place, and its label says so (`_rebound_label`)."""
    key = f"bilateral_option:{rule.source.composite}:" + identity.model_dump_json(exclude_none=True)
    return facts.remembered(key, lambda: _read_bilateral_option(identity, rule, facts))


def _read_bilateral_option(
    identity: BaseIdentity, rule: DestinationRule, facts: ScopeFacts
) -> Option:
    source = rule.source
    binding = ProviderBinding(scope="bilateral", provenance=source.composite)
    if rule.multilateral is not None:
        return _option(
            "bilateral",
            _unavailable("multilateral_destination", {"destination": rule.multilateral}),
            binding=binding,
            label=source.label,
        )
    if _population_unavailable(source.composite, facts):
        return _option(
            "bilateral",
            _unavailable("provider_not_offered", {"why": "destination_unclassified"}),
            binding=binding,
            label=source.label,
        )
    try:
        years = _composite_years(identity, rule.read(identity), source.composite, facts)
    except CompositeNotServable as exc:
        return _option(
            "bilateral",
            _composite_fault(identity, exc, uncovered="option_not_covered"),
            binding=binding,
            label=source.label,
        )
    if identity.requested_window.kind == "latest":
        own_year = elected_year(_year_counts(identity, facts))
        option = _latest_sibling_option(
            "bilateral", binding, years.own, years.held, years.counts, own_year, label=source.label
        )
    else:
        option = _sibling_option("bilateral", binding, years.own, years.held, label=source.label)
    if rule.spanning is None:
        return option
    return option.model_copy(update={"label": _rebound_label(option.label, source)})


@dataclass(frozen=True, slots=True)
class _CompositeYears:
    """What the bilateral option of a part reads (`_composite_years`), for each donor the
    part names, in order, or for every donor together where it names none: `own`, the years
    it holds a series of the total the part counts, and `held`, the years the composite
    serves it (`composite.served_components`). `counts` are the composite's series per year
    it serves (`_served_counts`), which `elect_latest_window` elects by too."""

    own: list[frozenset[int]]
    held: list[frozenset[int]]
    counts: dict[int, int]


def _composite_years(
    identity: BaseIdentity, read: BaseIdentity, composite: str, facts: ScopeFacts
) -> _CompositeYears:
    """The years the part whose identity is `identity` holds of its total, and those
    `composite` over `read` serves it, from the components it binds
    (`composite.served_components`): a population's members on an open destination, so a
    donor whose rows sit at the DAC's totals alone holds none. Raises
    `code_index.composites.CompositeNotServable` as `composite.served_components` does."""
    document = _with_composite(read.model_dump(mode="json", exclude_none=True), composite)
    served = served_components(document, facts=facts)
    covered = frozenset[int].intersection(*(c.served for c in served)) if served else frozenset()
    counts = _served_counts(served)
    donors = identity.entities.get("origin") or []
    if not donors:
        own = frozenset(y for y, n in _year_counts(identity, facts).items() if n > 0)
        return _CompositeYears(own=[own], held=[covered], counts=counts)
    by_component = [donor_years(c.component.identity, facts=facts) for c in served]
    held = [
        covered & frozenset[int]().union(*(years.get(donor, frozenset()) for years in by_component))
        for donor in donors
    ]
    return _CompositeYears(own=_donor_years(identity, facts), held=held, counts=counts)


def _rebound_label(label: str, source: BilateralSource) -> str:
    """`label` read at `source`'s total of the bilateral side in place of its total of both
    sides (`OPTION_LABELS.provider_rebound`)."""
    return OPTION_LABELS.provider_rebound.format(
        label=label, rebound=source.rebound_words, spanning=source.spanning_words
    )


def _published_inclusive(
    row: PublishedProviderScope,
    source: InclusiveSource | None,
    bilateral: BaseIdentity,
    facts: ScopeFacts,
    *,
    held: bool,
    rule: DestinationRule | None,
) -> Option:
    """The inclusive option of the published total `row` whose identity without a provider
    composite is `bilateral` (`_published_table`), whose destinations read by `rule` against
    the bilateral source of `row`'s total, None where it declares none. The inclusive
    composite's bilateral line reads the bilateral destinations alone, so the option is
    unavailable (`multilateral_destination`) on a part naming a multilateral destination, not
    offered (`provider_not_offered`, why `destination_unclassified`) where its population is
    unavailable, as the bilateral option is, and on a part naming the total of both sides it
    reads the total of the bilateral side in its place, as its label says
    (`_rebound_label`)."""
    if source is None:
        if row.scope in _COUNTS_CONTRIBUTIONS:
            return _option(
                "inclusive", _unavailable("contributions_counted", {"dataset": row.dataset})
            )
        return _option("inclusive", _unavailable("provider_not_offered", {"why": "dataset"}))
    if rule is None:
        return _inclusive_option(bilateral, source, facts, held=held)
    if rule.multilateral is not None:
        return _option(
            "inclusive",
            _unavailable("multilateral_destination", {"destination": rule.multilateral}),
            source=source.composite,
        )
    if _population_unavailable(source.composite, facts):
        return _option(
            "inclusive",
            _unavailable("provider_not_offered", {"why": "destination_unclassified"}),
            source=source.composite,
        )
    option = _inclusive_option(rule.read(bilateral), source, facts, held=held)
    if rule.spanning is None:
        return option
    return option.model_copy(update={"label": _rebound_label(option.label, rule.source)})


def with_bilateral_default(identity: BaseIdentity, facts: ScopeFacts) -> BaseIdentity:
    """`identity` bound to the composite of the bilateral source (`BILATERAL_SOURCES`, rule
    `BilateralSource.rule`) of the published total it counts, where a fresh search reads it:
    `identity` binds the total's code as a value on a described shape, binds no source but a
    pin to the total's dataset, the index elects that dataset for it, it names no
    destination or only destinations on the bilateral side (`destination_rule`), and its
    bilateral option is available (`_bilateral_option`), so the composite serves every year
    its answer holds. A donor whose total ODA holds no breakdown by recipient in those years
    keeps total ODA, and so does a part whose population is unavailable. The donors it names
    do not matter otherwise, so the rule holds whether or not the part has a provider axis,
    as for a ranking over every donor. `identity` itself otherwise.

    The composite then is the answer's own reading and the scope's base: its provider table
    holds the bilateral option, a switch of another axis posted over the base keeps it, and a
    switch to the total itself or to the inclusive scope reads another source. The composite
    replaces a provenance pin to the total's dataset (`edits._with_source`), so a part
    pinned to DAC2a and switched to total ODA from the composite reads the unpinned total,
    whose figures are the pinned one's under another spec_id. A resubmit settles nothing,
    so a posted identity reads as it binds: a bare total holds the contributions scope
    (`PUBLISHED_PROVIDER_SCOPES`)."""
    defaulted = _default_bilateral_option(identity, facts)
    if defaulted is None or defaulted[1].availability.status != "available":
        return identity
    source, _ = defaulted
    return apply_binding(
        identity, ProviderBinding(scope="bilateral", provenance=source.composite), []
    )


def _default_bilateral_option(
    identity: BaseIdentity, facts: ScopeFacts
) -> tuple[BilateralSource, Option] | None:
    """The bilateral source of the published total `identity` counts and its bilateral
    option (`_bilateral_option`), where the default reaches `identity`
    (`with_bilateral_default`) whether or not the option is available. None otherwise."""
    reached = _default_rule(identity, facts)
    if reached is None:
        return None
    return reached.source, _bilateral_option(identity, reached, facts)


def counted_source(identity: BaseIdentity, index: CodeIndex) -> BilateralSource | None:
    """The bilateral source whose published total `identity`'s bound slots count
    (`edits.counted_bilateral_source`), declared for the dataset `index` elects for
    `identity`'s family: the one rule deciding whether the bilateral default can reach a
    reading, read by the default itself (`_default_rule`) and by the election that settles
    a part's provider reading before its read (`reading.elect_reading`). None where no
    source is counted."""
    return counted_bilateral_source(identity, dataset_of=lambda held: _dataset(held, index))


def bilateral_default_reaches(identity: BaseIdentity, facts: ScopeFacts) -> bool:
    """Whether the bilateral default reaches `identity` (`with_bilateral_default`), whether
    or not its option is available. It reads the code index alone, never coverage."""
    return _default_rule(identity, facts) is not None


def _default_rule(identity: BaseIdentity, facts: ScopeFacts) -> DestinationRule | None:
    """The destination rule of the published total `identity` counts against its bilateral
    source, where the default reaches `identity`: it binds a source's code, reads its
    dataset under no other pin, and names no destination off the bilateral side. None
    otherwise."""
    if not isinstance(identity.shape, IdentityShapeSchema):
        return None
    source = counted_source(identity, facts.index)
    if source is None:
        return None
    pin = (identity.facets or {}).get("provenance")
    if pin is not None and pin != IdentityBindingValue(ref=source.dataset):
        return None
    rule = destination_rule(identity, source, facts)
    return rule if rule.bilateral else None


def bilateral_default_withheld(
    identity: BaseIdentity, selection: ScopeState, facts: ScopeFacts
) -> Withheld | None:
    """Why a fresh reading of `identity` that the bilateral default reaches
    (`with_bilateral_default`) reads total ODA, where no provider axis states it
    (`provider_rule`) and no stated or carried provider entry asks for total ODA: the
    reason its unavailable bilateral option states (`_bilateral_option`, `_withheld`),
    `destination_unclassified` for an unavailable population, `slot_not_served` naming the
    slot where the composite cannot serve one `identity` binds, and `option_not_covered` where
    it does not cover every year the answer holds. A part with a provider axis states the reason on its
    bilateral option instead. None where the default applies or does not reach `identity`."""
    defaulted = _default_bilateral_option(identity, facts)
    if defaulted is None:
        return None
    availability = defaulted[1].availability
    if availability.status == "available":
        return None
    stated = selection.entries_on("provider")
    if stated and stated[0].option_id == CONTRIBUTIONS:
        return None
    if provider_rule(identity, None, facts) is not None:
        return None
    return _withheld(availability)


def _withheld(availability: OptionAvailability) -> Withheld:
    """The reason an unavailable or partial bilateral option states (`_read_bilateral_option`):
    `destination_unclassified` or `slot_not_served`, with its slot, from a
    `provider_not_offered` detail, and `option_not_covered` from that code. Any other state
    is `scope_undecided`, logged as an error: the fallback for an unreachable state,
    since the default does not reach a part naming a multilateral destination
    (`multilateral_destination`), and `composite_not_servable` has no registered reason."""
    detail = availability.detail
    match availability.code, detail.get("why"):
        case "provider_not_offered", "destination_unclassified":
            return Withheld(why="destination_unclassified")
        case "provider_not_offered", "slot_not_served":
            return Withheld(why="slot_not_served", slot=detail["slot"])
        case "option_not_covered", None:
            return Withheld(why="option_not_covered")
    logger.error(
        "bilateral option states no registered withheld reason: %s %s",
        availability.code,
        detail,
    )
    return Withheld(why="scope_undecided")


def bilateral_default_reading(
    identity: BaseIdentity, selection: ScopeState, facts: ScopeFacts
) -> tuple[BaseIdentity, ScopeState]:
    """The identity a fresh reading of `identity` is served as, and the selection its scope
    reads: the one rule the reading elected before a part's read (`reading.elect_reading`)
    and the reading elected at the read of each candidate, refusal alternate, flagged match
    and nearest-real suggestion (`reading.elect_at_read`) both apply, through
    `scope_decision.choose_provider`.

    DAC2a's total ODA reads its bilateral destinations (`with_bilateral_default`) unless the
    part's first provider entry, explicit before carried (`ScopeState.entries_on`), asks for
    total ODA (`provider:contributions`), which keeps `identity`. Where the part has a
    provider axis (`provider_rule`), its table settles the entry as it settles any
    (`settle_provider`). Where it has none, as on a ranking over every donor, no table can
    apply the entry, so the default serves it here: a `provider:contributions` entry keeps
    total ODA and a `provider:bilateral` one reads the composite, and the entry leaves the
    selection, so the scope does not report it as dropped. Any other entry, or one on a part
    the default does not reach, stays in the selection as it was posted."""
    default = with_bilateral_default(identity, facts)
    stated = selection.entries_on("provider")
    if default is identity or not stated or stated[0].option_id not in (CONTRIBUTIONS, BILATERAL):
        return default, selection
    entry = stated[0]
    served = identity if entry.option_id == CONTRIBUTIONS else default
    if provider_rule(served, None, facts) is not None:
        return served, selection
    entries = tuple(each for each in selection.entries if each is not entry)
    return served, dataclasses.replace(selection, entries=entries)


def _sibling_option(
    scope: ProviderScope,
    binding: SlotBinding | ProviderBinding,
    own: Sequence[frozenset[int]],
    held: Sequence[frozenset[int]],
    *,
    label: str | None = None,
) -> Option:
    """The option of `scope` binding the sibling total `binding` names, labelled `label` or
    else its scope's label, on a part asking for a range or every year. `own` and `held` name, for each donor of the part in order,
    the years it holds the part's own total and the sibling in (`_donor_years`). The answer
    holds the years any donor holds its own total in; one of them is covered when every
    donor holding its own total in it holds the sibling in it, so a donor holding no own
    total in a year places no demand on it. The option is available when every year the
    answer holds is covered, partial over the covered years when those are fewer, with
    their latest run as the window a switch reads (`SlotBinding.window`) and the years its
    label states, and unavailable when none is. Where no donor holds a year of its own
    total, the answer holds no year to measure against, so the option is available where
    every donor holds the sibling in the window, and unavailable otherwise. Each reason is
    `option_not_covered`."""
    asked = frozenset[int]().union(*own)
    if not asked:
        return (
            _option(scope, binding=binding, label=label)
            if all(held)
            else _option(scope, _unavailable("option_not_covered"), binding=binding, label=label)
        )
    covered = frozenset(
        year
        for year in asked
        if all(year in theirs for ours, theirs in zip(own, held, strict=True) if year in ours)
    )
    if covered >= asked:
        return _option(scope, binding=binding, label=label)
    if not covered:
        return _option(scope, _unavailable("option_not_covered"), binding=binding, label=label)
    runs = spans(covered)
    return _option(
        scope,
        OptionAvailability(status="partial", years=runs, code="option_not_covered"),
        binding=binding.model_copy(update={"window": runs[-1]}),
        window=runs[-1],
        label=label,
    )


def _latest_sibling_option(
    scope: ProviderScope,
    binding: SlotBinding | ProviderBinding,
    own: Sequence[frozenset[int]],
    held: Sequence[frozenset[int]],
    counts: Mapping[int, int],
    own_year: int | None,
    *,
    label: str | None = None,
) -> Option:
    """The option of `scope` binding the sibling total `binding` names, labelled `label` or
    else its scope's label, on a part asking for the latest year, whose read elects `own_year` for the pool. `own` and `held` are as
    `_sibling_option` takes them, and `counts` the sibling's series per year
    (`_year_counts`). A `latest` read elects one year for the whole pool, so the option
    elects the sibling's year as `elected_year` does among the years every donor holds it
    in, the donors holding their own total in `own_year` (every donor when it is None),
    and a switch reads that one year (`SlotBinding.window`). The option is available when
    that year is `own_year`, partial over it, stated in its label, when it is another, and
    unavailable when the donors share no year, or when no donor holds its own total in
    `own_year`, which the series count and the donor scan can disagree on. Each reason is
    `option_not_covered`."""
    demanded = [
        theirs
        for ours, theirs in zip(own, held, strict=True)
        if own_year is None or own_year in ours
    ]
    shared = frozenset[int].intersection(*demanded) if demanded else frozenset[int]()
    year = elected_year({y: n for y, n in counts.items() if y in shared})
    if year is None:
        return _option(scope, _unavailable("option_not_covered"), binding=binding, label=label)
    window = (year, year)
    pinned = binding.model_copy(update={"window": window})
    if year == own_year:
        return _option(scope, binding=pinned, label=label)
    return _option(
        scope,
        OptionAvailability(status="partial", years=[window], code="option_not_covered"),
        binding=pinned,
        window=window,
        label=label,
    )


def _tables(identity: BaseIdentity, facts: ScopeFacts) -> list[Mapping[str, object]]:
    """The identity of each table `identity` reads: its components on the OECD headline
    (`component_identities`), itself otherwise."""
    document = identity.model_dump(mode="json", exclude_none=True)
    components = component_identities(facts.index, document)
    return [document] if components is None else [c.identity for c in components]


def _year_counts(identity: BaseIdentity, facts: ScopeFacts) -> dict[int, int]:
    """How many series of the tables `identity` reads hold each year of its window
    (`component_years`), summed over the tables."""
    counts: dict[int, int] = {}
    for table in _tables(identity, facts):
        for year, series in component_years(table, facts=facts).counts.items():
            counts[year] = counts.get(year, 0) + series
    return counts


def _donor_years(identity: BaseIdentity, facts: ScopeFacts) -> list[frozenset[int]]:
    """The years of its window in which each donor `identity` names holds a series of the
    total it counts, on its basis, in the order it names them, from one coverage scan per
    table it reads (`composite.donor_years`). On the OECD headline, a year counts on the
    basis the headline reads it on: net flows before 2018, grant equivalents from 2018
    (`component_identities`)."""
    held = [donor_years(table, facts=facts) for table in _tables(identity, facts)]
    return [
        frozenset[int]().union(*(years.get(donor, frozenset()) for years in held))
        for donor in identity.entities["origin"]
    ]


def _option(
    scope: ProviderScope,
    availability: OptionAvailability | None = None,
    *,
    binding: OptionBinding | None = None,
    window: YearSpan | None = None,
    source: str | None = None,
    label: str | None = None,
) -> Option:
    """The option of `scope`, with `availability`, available when None, binding `binding` or
    else the scope itself over `window` with the composite `source`, labelled `label` or else
    its scope's label. A `window` also states its years in the label
    (`OPTION_LABELS.provider_years`): the years both lines of the inclusive scope cover, or
    the run a partial sibling total serves."""
    named = label or _SCOPE_LABELS[scope]
    return Option(
        option_id=provider_option_id(scope),
        label=(
            named
            if window is None
            else OPTION_LABELS.provider_years.format(label=named, years=_years_label(window))
        ),
        binding=(
            binding
            if binding is not None
            else ProviderBinding(scope=scope, window=window, provenance=source)
        ),
        availability=availability or OptionAvailability(status="available"),
    )


def _unavailable(
    code: ScopeReasonCode, detail: Mapping[str, str] | None = None
) -> OptionAvailability:
    """An unavailable option's availability, for `code` with `detail`."""
    return OptionAvailability(status="unavailable", code=code, detail=dict(detail or {}))


def _published_row(identity: BaseIdentity, facts: ScopeFacts) -> PublishedProviderScope | None:
    """The declared row (`PUBLISHED_PROVIDER_SCOPES`) of the total `identity` counts: a row
    whose code `identity` binds as a value and whose dataset the index elects for the table
    identity `identity` reads (`table_identity`, the net identity of the OECD headline), when
    every provider `identity` names counts as a donor (`_counts_as_donor`) and they do not
    pool a redistributed provider with one of its funders (`_pools_funder`). None otherwise,
    or when the composite `identity` binds cannot derive its components. The bound codes
    and the providers are read before the family lookup.

    A pool of an EU member and the EU institutions counts the member's core contributions to
    the institutions twice in total ODA, as the member's contributions and as the
    institutions' own aid, and a switch from bilateral ODA would reach that total, so such a
    pool has no record: every scope a record would hold or offer for it is either one it
    double counts or one a switch leaves for one it double counts. A pool of the EU
    institutions and a donor outside the EU counts no contribution twice. Membership is
    read over the part's window (`_window_days`), so the United Kingdom counts as a member
    over any window holding a year before it left in 2020."""
    bound = {
        (slot.property, slot.binding.ref)
        for slot in identity.slots
        if isinstance(slot.binding, IdentityBindingValue)
    }
    rows = [row for row in PUBLISHED_PROVIDER_SCOPES if (row.code_system, row.code) in bound]
    if not rows or not _names_donors_alone(identity, facts.index):
        return None
    try:
        document = table_identity(
            facts.index, identity.model_dump(mode="json", exclude_none=True)
        ).identity
    except CompositeNotServable:
        return None
    dataset = _dataset(BaseIdentity.model_validate(document), facts.index)
    return next((row for row in rows if row.dataset == dataset), None)


def provider_total_code(code_system: str, code: str, index: CodeIndex) -> str | None:
    """The published total a provider option on `code_system` reads `code` as a scope of: the
    outermost code at or above `code` (`CodeRow.parent`) that `PUBLISHED_PROVIDER_SCOPES`
    declares on `code_system`. DAC1's bilateral ODA (1015) and multilateral ODA (2000) read
    total ODA (1010). None where no row declares `code` on `code_system`."""
    declared = {row.code for row in PUBLISHED_PROVIDER_SCOPES if row.code_system == code_system}
    if code not in declared:
        return None
    total = code
    while (row := index.code_by_dcid(total)) is not None and row.parent is not None:
        if row.parent not in declared:
            break
        total = row.parent
    return total


def _names_donors_alone(identity: BaseIdentity, index: CodeIndex) -> bool:
    """Whether `identity` names one or more providers, every one a donor
    (`_counts_as_donor`), with no redistributed provider pooled with one of its funders
    (`_pools_funder`): the providers whose published total and imputed share count no
    multilateral's outflows twice."""
    providers = identity.entities.get("origin") or []
    return (
        bool(providers)
        and not _pools_funder(providers, identity.requested_window, index)
        and all(_counts_as_donor(provider, index) for provider in providers)
    )


def without_uncounted_inclusive(
    identity: BaseIdentity, entries: Sequence[Entry] | None, *, facts: ScopeFacts
) -> tuple[BaseIdentity, Sequence[Entry] | None]:
    """A posted `identity` and its scope `entries`, with the inclusive scope its identity
    binds read as a carried entry where a fresh search would not offer it.

    The rule is the one a fresh search reads: the provider table of the identity's
    bilateral line (`provider_rule`) offers the inclusive option, so the providers it names
    can take the imputed share (`_inclusive_option`, and on a published total
    `_published_row`). Where it does not, as for a part naming no provider, or a provider
    whose total already counts the multilaterals' outflows, the identity is served its
    bilateral line: the table the composite stacks on (`_without_provider_composite`), read
    as a fresh search reads it (`with_bilateral_default`), so DAC2a's total ODA reads the
    bilateral destinations alone. `provider:inclusive` joins its entries as a carried entry
    from the posted identity's provider choice, unless they hold a provider entry, so the
    scope drops it as it drops any carried entry the part cannot take (`compile`), and says
    so. `identity` and `entries` themselves otherwise."""
    try:
        ref = composite_ref(identity.model_dump(mode="json", include={"facets"}))
    except CompositeNotServable:
        return identity, entries
    if ref not in _SOURCE_COMPOSITES:
        return identity, entries
    bilateral = _without_provider_composite(identity, ref)
    table = provider_rule(bilateral, None, facts)
    option = table.option(INCLUSIVE) if table is not None else None
    if option is not None and option.availability.code != "provider_not_offered":
        return identity, entries
    held = list(entries or ())
    if not any(entry.axis == "provider" for entry in held):
        held.append(
            Entry(
                axis="provider",
                option_id=INCLUSIVE,
                provenance="carried",
                origin=ChoiceOrigin(choice_id=f"{identity_spec_id(identity)}:provider"),
            )
        )
    return with_bilateral_default(bilateral, facts), held


def _counts_as_donor(party: str, index: CodeIndex) -> bool:
    """Whether a published donor total counts `party` as a donor: a country, an organisation
    `DONOR_ORGANISATIONS` declares, or a total of a group of countries alone
    (`_members`), as the G7 or the DAC countries. Any other organisation or total, the
    DAC's total of the multilaterals among them, counts multilateral outflows."""
    if party in _DONOR_ORGANISATIONS:
        return True
    members = _members(party, index)
    return bool(members) and all(member.startswith(COUNTRY_PREFIX) for member in members)


def _members(party: str, index: CodeIndex) -> frozenset[str]:
    """What a total of `party` sums: the country itself, or the members of the group it
    totals (`PartyRow.totals`) in the membership policy (`region.policy_members`); empty for
    any other party."""
    if party.startswith(COUNTRY_PREFIX):
        return frozenset((party,))
    row = index.parties.get(party)
    if row is None or row.totals is None:
        return frozenset()
    return policy_members(row.totals)


def _pools_funder(providers: Collection[str], window: IdentityWindow, index: CodeIndex) -> bool:
    """Whether `providers` pool a redistributed provider (`REDISTRIBUTED_PROVIDERS`) with a
    member of the group that funds it on any day of `window` (`_window_days`), or a total
    that sums one (`_members`), such as the EU institutions with an EU member state or with
    the G7. A published total counts such a pool's core contributions to the provider twice. Where
    the policy holds no members for the funding group, any pool with the provider counts as
    one."""
    first, last = _window_days(window)
    for row in REDISTRIBUTED_PROVIDERS:
        if row.provider not in providers:
            continue
        funders = policy_members_between(row.funders, first, last)
        others = (provider for provider in providers if provider != row.provider)
        if any(not funders or funders & _members(other, index) for other in others):
            return True
    return False


def _window_days(window: IdentityWindow) -> tuple[date, date]:
    """The first and last days `window` reads: a range's years, open where it states no
    bound, every day for every year, and today for the latest year, which reads under
    today's memberships."""
    match window:
        case IdentityWindowRange(start_year=start, end_year=end):
            return (
                date.min if start is None else date(start, 1, 1),
                date.max if end is None else date(end, 12, 31),
            )
        case IdentityWindowLatest():
            today = date.today()
            return today, today
        case _:
            return date.min, date.max


def _pools_redistributed(providers: Collection[str]) -> bool:
    """Whether `providers` pool two or more providers, one of them a multilateral whose own
    outflows a scope counting multilateral aid also counts inside its funders' shares
    (`REDISTRIBUTED_PROVIDERS`)."""
    return len(providers) > 1 and not _REDISTRIBUTED.isdisjoint(providers)


def _dataset(identity: BaseIdentity, index: CodeIndex) -> str | None:
    """The dataset the index elects for `identity`'s family, None when it decides none."""
    try:
        family = index_family(
            index, identity_family_query(index, identity.model_dump(mode="json", exclude_none=True))
        )
    except SourceNotInFamily:
        return None
    return family.provenance if family is not None else None


def _inclusive_option(
    bilateral: BaseIdentity, source: InclusiveSource, facts: ScopeFacts, *, held: bool
) -> Option:
    """The inclusive option binding `source`'s composite over `bilateral`, the part's
    identity without it. It is not offered for what `_not_offered` and
    `_providers_not_offered` refuse, or where the imputed line cannot serve a slot
    `bilateral` binds (why `slot_not_served`, `_composite_fault`), even when the part holds
    it. For a part asking for the latest year it
    is available over the year `inclusive_reach` elects, which its window states
    (`elect_latest_window`). Otherwise it is available when both lines cover every year the
    bilateral answer holds; partial over the latest run of years they cover, with that run as
    its window, when they cover fewer; and unavailable when they cover none. The option a part
    holds over a range or every year is available over its own window. On a part counted on
    net flows its label says the imputed line is added as published
    (`OPTION_LABELS.provider_inclusive_net`)."""
    try:
        why = _not_offered(bilateral, source) or _providers_not_offered(bilateral, source, facts)
    except CompositeNotServable as exc:
        return _option("inclusive", _composite_fault(bilateral, exc, uncovered="imputed_coverage"))
    if why is not None:
        return _option("inclusive", _unavailable("provider_not_offered", {"why": why}))
    measure = held_measure(bilateral)
    label = (
        OPTION_LABELS.provider_inclusive_net
        if measure is not None and measure.basis == _NET
        else None
    )

    def option(
        availability: OptionAvailability | None = None, window: YearSpan | None = None
    ) -> Option:
        return _option(
            "inclusive", availability, window=window, source=source.composite, label=label
        )

    latest = bilateral.requested_window.kind == "latest"
    if held and not latest:
        return option()
    reach = inclusive_reach(bilateral, facts, composite=source.composite)
    if latest:
        if reach.latest is None:
            return option(_unavailable("imputed_coverage"))
        return option(window=(reach.latest, reach.latest))
    if not reach.covered:
        return option(_unavailable("imputed_coverage"))
    if reach.covered >= reach.asked:
        return option()
    runs = spans(reach.covered)
    return option(
        OptionAvailability(status="partial", years=runs, code="imputed_coverage"), runs[-1]
    )


def elect_latest_window(identity: BaseIdentity, facts: ScopeFacts) -> BaseIdentity:
    """`identity`, served as it reads: a composite a provider option binds asking for the
    latest year is read over the year `inclusive_reach` elects among those its lines cover, a
    one-year range, since its lines can elect different latest years and the Datasets mint
    refuses a composite's `latest` window. The bilateral destinations of DAC2a's total ODA
    (`BILATERAL_SOURCES`) are one line, read over the latest year it serves. Any other
    identity, or one whose tables elect no year, is `identity` itself.

    A part's base keeps its `latest` window, so an option that removes the composite
    compiles to its table's own latest year; a composite compiles here to one year."""
    if identity.requested_window.kind != "latest":
        return identity
    try:
        ref = composite_ref(identity.model_dump(mode="json", include={"facets"}))
    except CompositeNotServable:
        return identity
    if ref not in _PROVIDER_COMPOSITES:
        return identity
    try:
        year = inclusive_reach(
            _without_provider_composite(identity, ref), facts, composite=ref
        ).latest
    except CompositeNotServable:
        return identity
    if year is None:
        return identity
    return identity.model_copy(
        update={"requested_window": IdentityWindowRange(start_year=year, end_year=year)}
    )


def _years_label(window: YearSpan) -> str:
    first, last = window
    return str(first) if first == last else f"{first} to {last}"


def _not_offered(identity: BaseIdentity, source: InclusiveSource) -> str | None:
    """Why the inclusive scope binding `source`'s composite cannot serve `identity`, None
    when it can: the identity pins a provenance that admits a table other than the one the
    bilateral answer reads (`provenance_pinned`; a pin to `InclusiveSource.dataset` alone
    passes, since the composite reads that table), is counted on a basis the source does not
    add its imputed line on (`basis`, `InclusiveSource.bases`), or at constant prices, which
    the imputed lines are not published at (`constant_prices`)."""
    pin = (identity.facets or {}).get("provenance")
    if pin is not None and _pinned_tables(pin) != {source.dataset}:
        return "provenance_pinned"
    measure = held_measure(identity)
    if measure is None or measure.basis not in source.bases:
        return "basis"
    shape = identity.shape
    unit = (identity.facets or {}).get("unit")
    if (isinstance(shape, IdentityShapeSchema) and shape.measurement_qualifier == _CONSTANT) or (
        isinstance(unit, IdentityFacetValueRequest) and unit.request.prices != "current"
    ):
        return "constant_prices"
    return None


def _pinned_tables(pin: object) -> set[str]:
    """The tables a provenance binding admits: its value, or the members of its set. Empty
    for any other binding, which admits every table."""
    if isinstance(pin, IdentityBindingValue):
        return {pin.ref}
    if isinstance(pin, IdentityBindingSet):
        return set(pin.refs)
    return set()


def _providers_not_offered(
    identity: BaseIdentity, source: InclusiveSource, facts: ScopeFacts
) -> str | None:
    """Why the inclusive scope binding `source`'s composite cannot count the providers
    `identity` names, None when it can.

    The imputed line attributes each multilateral's spending to the providers that fund it,
    each by its own share of the multilateral's core resources, so the imputed shares of two
    providers are disjoint and add. The scope serves a part when the table carries rows for
    every provider it names, in the family the scope reads, in any year and to any recipient,
    and, for two or more providers, none is a multilateral whose own outflows the table's rows
    attribute to others (`REDISTRIBUTED_PROVIDERS`). A pool holding a multilateral the table
    carries no rows for, IDA beside the US, would add IDA's outflows to the US share of them.
    Otherwise the why is `all_providers` for a part naming no provider, whose total never adds
    imputed figures, `provider` for one provider the table does not carry, and
    `pooled_providers` for several. Raises `code_index.composites.CompositeNotServable` where
    the imputed table cannot serve what `identity` binds."""
    providers = identity.entities.get("origin") or []
    if not providers:
        return "all_providers"
    document = identity.model_dump(mode="json", exclude_none=True)
    components = component_identities(facts.index, _with_composite(document, source.composite))
    assert components is not None
    imputed = next(c.identity for c in components if c.role == "imputed")
    carried = all(
        component_years(
            {**imputed, "entities": {"origin": [provider]}, "requested_window": {"kind": "all"}},
            facts=facts,
        ).held
        for provider in providers
    )
    if len(providers) == 1:
        return None if carried else "provider"
    return None if carried and not _pools_redistributed(providers) else "pooled_providers"


def _composite_fault(
    identity: BaseIdentity, exc: CompositeNotServable, *, uncovered: ScopeReasonCode
) -> OptionAvailability:
    """The availability of a provider option whose composite refuses what `identity` binds
    (`exc`): not offered (`provider_not_offered`) with why `destination_unclassified` for an
    unavailable population (`PopulationUnavailable`) and why `slot_not_served`, its `slot`
    the property at fault, for a slot the composite's components do not declare; unavailable
    with `uncovered`, the option's code for a composite serving no year, where the window
    reaches none of the composite's component periods. At any other member (`/shape`,
    `/entities/<key>`, a provenance set or a composite the index does not serve) it is not
    offered with why `composite_not_servable` and the `pointer` at fault, so one option's
    refusal never aborts the scope build."""
    if isinstance(exc, PopulationUnavailable):
        return _unavailable("provider_not_offered", {"why": "destination_unclassified"})
    if exc.pointer == "/requested_window":
        return _unavailable(uncovered)
    slot = _slot_at(identity, exc.pointer)
    if slot is None:
        return _unavailable(
            "provider_not_offered", {"why": _COMPOSITE_NOT_SERVED, "pointer": exc.pointer}
        )
    return _unavailable("provider_not_offered", {"why": _SLOT_NOT_SERVED, "slot": slot})


def _slot_at(identity: BaseIdentity, pointer: str) -> str | None:
    """The slot property at `pointer` (`/slots/<n>`) of `identity`, None at any other
    member."""
    head, _, position = pointer.rpartition("/")
    if head == "/slots" and position.isdigit() and int(position) < len(identity.slots):
        return identity.slots[int(position)].property
    return None


def provider_convention(
    identity: BaseIdentity, facts: ScopeFacts, *, asks_total: bool
) -> ProviderConvention | None:
    """The declared convention (`PROVIDER_CONVENTIONS`) that matches the part whose bilateral
    identity is `identity` and whose words ask for a total when `asks_total`, None when none
    does. Read once per scope build for `identity` and `asks_total`."""
    key = f"provider_convention:{asks_total}:" + identity.model_dump_json(exclude_none=True)
    return facts.remembered(key, lambda: _convention(identity, facts, asks_total=asks_total))


def _convention(
    identity: BaseIdentity, facts: ScopeFacts, *, asks_total: bool
) -> ProviderConvention | None:
    """The convention `provider_convention` reads."""
    shape = identity_part_shape(identity, facts.index)
    names_donor = bool(identity.entities.get("origin"))
    dataset = _dataset(identity, facts.index)
    return next(
        (
            row
            for row in PROVIDER_CONVENTIONS
            if shape in row.shapes
            and row.dataset == dataset
            and (names_donor or not row.names_donor)
            and (asks_total or not row.asks_total)
        ),
        None,
    )


def with_summed_provider(
    identity: BaseIdentity, state: ScopeState, facts: ScopeFacts
) -> ScopeState:
    """`state` with its words summing the bilateral and multilateral lines
    (`ScopeState.sums_providers`, "bilateral plus multilateral ODA") read as the option
    summing them on the dataset of the part whose answer is `identity`. Where the part's
    provider table offers the contributions scope (`provider_rule`) on a dataset that
    declares no inclusive source (`INCLUSIVE_SOURCES`), DAC1's total ODA, which counts
    bilateral flows plus core contributions to multilaterals, the explicit inclusive entry
    read from those words asks for that scope. Where the dataset declares one, on the CRS
    and DAC2a, the inclusive composite is the sum, whatever destination the part names, and
    `state` is itself."""
    if not state.sums_providers:
        return state
    table = provider_rule(identity, None, facts)
    if table is None or table.option(CONTRIBUTIONS) is None:
        return state
    try:
        ref = composite_ref(identity.model_dump(mode="json", include={"facets"}))
    except CompositeNotServable:
        return state
    if _dataset(_without_provider_composite(identity, ref), facts.index) in _SOURCES:
        return state
    entries = tuple(
        entry.model_copy(update={"option_id": CONTRIBUTIONS})
        if entry.option_id == INCLUSIVE
        and entry.provenance == "explicit"
        and isinstance(entry.origin, TextOrigin)
        else entry
        for entry in state.entries
    )
    return dataclasses.replace(state, entries=entries)


def without_unoffered_multilateral(
    identity: BaseIdentity, state: ScopeState, facts: ScopeFacts
) -> ScopeState:
    """`state` without the explicit multilateral entry its words state ("multilateral ODA",
    `stated_scope._MULTILATERAL_ROW`) where the provider table of the part whose answer is
    `identity` offers no multilateral option (`provider_rule`), as on a CRS or DAC2a part
    or one with no provider table. Those words name DAC1's multilateral ODA only on a
    dataset that publishes it, so elsewhere the axis is unstated and the entry is neither
    applied nor reported. `state` itself where the table offers the option."""
    table = provider_rule(identity, None, facts)
    if table is not None and table.option(MULTILATERAL) is not None:
        return state
    entries = tuple(
        entry
        for entry in state.entries
        if not (
            entry.option_id == MULTILATERAL
            and entry.provenance == "explicit"
            and isinstance(entry.origin, TextOrigin)
        )
    )
    if len(entries) == len(state.entries):
        return state
    return dataclasses.replace(state, entries=entries)


def settle_provider(
    identity: BaseIdentity, state: ScopeState, facts: ScopeFacts
) -> AxisRecord | None:
    """The provider record of a fresh search's part whose answer is `identity`, None where
    the axis does not apply (`provider_rule`).

    The first entry on the axis (`ScopeState.entries_on`, explicit before carried) whose
    option the pass can serve (`_served_in_pass`) settles it. An explicit entry on a partial
    option settles it too: the scope compiles the option over the latest run of years it
    covers (`ProviderBinding.window`), as a switch to it does, and the answer states the
    narrowing as a departure. A carried entry on a partial option is not applied, since
    only the user's own words or a switch accept fewer years than asked for. Otherwise the declared convention matching the part (`provider_convention`,
    with the request's total intent) settles its scope where the pass can serve it, else the
    scope `identity` holds, else the bilateral scope, else the first available option
    (`_settled`). Every convention reads the CRS, so a published total reads none and holds
    the scope its row declares. The record names the matched convention's rule when it holds
    the convention's scope, and otherwise the rule of the option it holds
    (`AxisTable.rule_of`)."""
    table = provider_rule(identity, None, facts)
    if table is None:
        return None
    if table.published:
        return _settled(table, state, None)
    ref = composite_ref(identity.model_dump(mode="json", include={"facets"}))
    bilateral = _without_provider_composite(identity, ref)
    convention = provider_convention(bilateral, facts, asks_total=state.total)
    return _settled(table, state, convention)


def _settled(
    table: AxisTable, state: ScopeState, convention: ProviderConvention | None
) -> AxisRecord:
    """The record of `table` under the matched `convention`: the first option an entry on
    the axis asks for that the pass serves, else the first of the convention's option, the
    held one and the bilateral one that it serves, else the first option `table` marks
    available, the bilateral one on a part the CRS serves. The record never holds an option
    `table` marks unavailable."""
    matched = provider_option_id(convention.scope) if convention is not None else None

    def rule(active: str) -> str:
        if convention is not None and active == matched:
            return convention.rule
        return table.rule_of(active)

    for entry in state.entries_on("provider"):
        if _served_in_pass(table, entry.option_id, explicit=entry.provenance == "explicit"):
            return AxisRecord(
                axis="provider",
                rule=rule(entry.option_id),
                active=entry.option_id,
                provenance=entry.provenance,
                entry=entry,
            )
    candidates = (table.held, BILATERAL) if matched is None else (matched, table.held, BILATERAL)
    active = next((o for o in candidates if _served_in_pass(table, o)), None)
    if active is None:
        active = next(o.option_id for o in table.options if o.availability.status == "available")
    return AxisRecord(axis="provider", rule=rule(active), active=active, provenance="default")


def _served_in_pass(table: AxisTable, option_id: str, *, explicit: bool = False) -> bool:
    """Whether a fresh search serves option `option_id` of `table` as its answer: it is
    the option the answer holds or one binding a composite, which the scope applies to the
    answer (`edits.applies_in_pass`, `build._with_recorded_composites`), and it is
    available or, asked for by an `explicit` entry, partial, served over its own window."""
    option = table.option(option_id)
    if option is None:
        return False
    status = option.availability.status
    if status != "available" and not (explicit and status == "partial"):
        return False
    return option_id == table.held or applies_in_pass(option.binding)
