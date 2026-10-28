"""The provider axis: whose spending a part counts, the donor's bilateral flows alone, with
its imputed share of multilateral spending, with its core contributions to multilaterals, or
the core contributions alone.

The inclusive scope binds a stacked composite of the part's bilateral flows and an imputed
multilateral line as the identity's source; the bilateral scope removes it. Each table that
offers the scope declares its composite and the measure bases it adds on
(`code_index.scope_conventions.INCLUSIVE_SOURCES`): the CRS with ONE's imputed multilateral
table on gross disbursements, and DAC2a's total ODA with its own imputed multilateral ODA on
gross and net flows. The axis is present on a part the CRS serves and on a donor's published
total (`provider_rule`). A published total holds the scope its declared row names
(`code_index.scope_conventions.PUBLISHED_PROVIDER_SCOPES`): DAC1's total ODA the
contributions scope, its bilateral ODA the bilateral scope and its multilateral ODA the
multilateral scope, and DAC2a's total ODA the bilateral scope. Each DAC1 total offers the
others' scopes as options binding their aid types. Such an option serves the years every donor
holding its own total in a year also holds the other in, or, for a part asking for the
latest year, the one year the pool's read elects (`_published_table`). Where a table declares
an inclusive source, the inclusive option serves the years both lines cover for the part
(`inclusive_reach`): available when they cover every year the bilateral answer holds, partial
with a shorter window of its own when they cover fewer, and, for a part asking for the latest
year, available over the year a `latest` read elects among those both cover. It is not
offered (`provider_not_offered`) where the imputed line cannot serve what the part binds, or
cannot count the providers it names (`_providers_not_offered`): a part naming no provider,
a provider the line carries no rows for, or a pool naming a multilateral whose outflows
the line attributes to its funders. On any other basis the source does not add on it
conflicts with the measure-basis option (`AxisTable.conflicts`).

The default is declared (`code_index.scope_conventions.PROVIDER_CONVENTIONS`): a named donor's
total of a CRS sector or recipient part counts both, when they cover every year asked for.
`settle_provider` applies it, or the part's explicit or carried entry, on the parsed door,
where the answer can serve it (`_served_in_pass`). A choice names the convention's rule only
where the request matched the convention and the part holds its scope. Otherwise it names
the rule of the option it holds, its own scope's (`AxisTable.rules`), as every resubmit
does.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import cast, get_args

from code_index.composites import (
    CompositeNotServable,
    component_identities,
    composite_ref,
    table_identity,
    without_composite,
)
from code_index.members import SourceNotInFamily, identity_family_query, index_family
from code_index.query import CodeIndex
from code_index.scope_conventions import (
    BASIS_MEASURES,
    DONOR_ORGANISATIONS,
    INCLUSIVE_SOURCES,
    OPTION_LABELS,
    PROVIDER_CONVENTIONS,
    PROVIDER_SCOPE_RULES,
    PUBLISHED_PROVIDER_SCOPES,
    REDISTRIBUTED_PROVIDERS,
    InclusiveSource,
    ProviderConvention,
    PublishedProviderScope,
)
from materialise.composite import spans
from search.catalogue.counts import COUNTRY_PREFIX
from search.engine.spec_id import identity_spec_id
from search.models import (
    BaseIdentity,
    ChoiceOrigin,
    Entry,
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
    YearSpan,
)
from search.scope.basis import held_measure, identity_part_shape
from search.scope.composite import component_years, donor_years, served_components
from search.scope.edits import applies_in_pass, apply_binding
from search.scope.region import policy_members, policy_members_between
from search.scope.state import AxisRecord, ScopeState
from search.scope.tables import AxisTable, Conflict, ScopeFacts
from search.scope.vocabulary import measure_basis_option_id, provider_option_id
from series_coverage.index import elected_year

BILATERAL = provider_option_id("bilateral")
"""The bilateral provider option."""
INCLUSIVE = provider_option_id("inclusive")
"""The inclusive provider option."""
_SCOPE_LABELS: Mapping[ProviderScope, str] = {
    "bilateral": OPTION_LABELS.provider_bilateral,
    "inclusive": OPTION_LABELS.provider_inclusive,
    "contributions": OPTION_LABELS.provider_contributions,
    "multilateral": OPTION_LABELS.provider_multilateral,
}

_SOURCES: Mapping[str, InclusiveSource] = {row.dataset: row for row in INCLUSIVE_SOURCES}
"""The inclusive source of each table that declares one, by the table's provenance."""
_SOURCE_COMPOSITES = frozenset(row.composite for row in INCLUSIVE_SOURCES)
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
_SLOT_WHY: Mapping[str, str] = {
    "crsCooperationModality": "modality",
    "crsTypeOfFinance": "finance_type",
}
"""The `why` of `provider_not_offered` for each slot the imputed table's rows do not declare."""


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
        counts = {year: sum(c.years.counts[year] for c in served) for year in covered}
        return InclusiveReach(
            asked=bilateral.years.held, covered=covered, latest=elected_year(counts)
        )
    return InclusiveReach(asked=bilateral.years.held, covered=covered)


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
    bilateral = _without_inclusive(base, ref)
    dataset = _dataset(bilateral, facts)
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


def _without_inclusive(identity: BaseIdentity, ref: str | None) -> BaseIdentity:
    """`identity`, whose composite is `ref`, without an inclusive source's composite: the
    identity its table serves its bilateral flows on. `identity` itself when `ref` is another
    one."""
    if ref not in _SOURCE_COMPOSITES:
        return identity
    return BaseIdentity.model_validate(
        without_composite(identity.model_dump(mode="json", exclude_none=True))
    )


def _published_table(
    row: PublishedProviderScope, identity: BaseIdentity, *, facts: ScopeFacts
) -> AxisTable:
    """The provider table of a part counting the published total `row` declares, whose
    identity is `identity` (`_published_row`).

    The part holds `row`'s scope, or the inclusive one where `identity` binds its dataset's
    inclusive source (`INCLUSIVE_SOURCES`). Every other row of its dataset and code system
    (`PUBLISHED_PROVIDER_SCOPES`) is an option binding that row's code, so a switch to it
    reads the total that counts its scope: on DAC1's total ODA, bilateral flows bind
    bilateral ODA and core contributions alone multilateral ODA, and on bilateral ODA, core
    contributions bind total ODA. Such an option serves the years every donor holding the
    part's total in a year also holds the sibling in (`_sibling_option`), or, on a part asking
    for the latest year, the one year the pool's read elects (`_latest_sibling_option`).
    Where the dataset declares an inclusive source, the inclusive option is offered as on a
    part the CRS serves (`_inclusive_option`), with its basis conflicts. Otherwise it is
    unavailable with the total's dataset: `contributions_counted` on a total counting core
    contributions, since it counts them in place of an imputed share of the multilaterals'
    spending, and `provider_not_offered` (why `dataset`) on any other. Each option names its
    row's rule, and the inclusive one its scope's. No preference but the held scope leads, so
    no default moves the part to another total."""
    rows = {
        sibling.scope: sibling
        for sibling in PUBLISHED_PROVIDER_SCOPES
        if (sibling.dataset, sibling.code_system) == (row.dataset, row.code_system)
    }
    source = _SOURCES.get(row.dataset)
    ref = composite_ref(identity.model_dump(mode="json", include={"facets"}))
    inclusive_held = source is not None and ref == source.composite
    bilateral = _without_inclusive(identity, ref)
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
        if sibling is None and scope != "inclusive":
            continue
        option_id = provider_option_id(scope)
        rules[option_id] = sibling.rule if sibling is not None else _SCOPE_RULES[scope]
        if sibling is None:
            options.append(_published_inclusive(row, source, bilateral, facts, held=inclusive_held))
            continue
        if sibling is row:
            options.append(_option(scope, binding=ProviderBinding(scope=scope)))
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
    held = INCLUSIVE if inclusive_held else provider_option_id(row.scope)
    return AxisTable(
        axis="provider",
        options=tuple(options),
        preference=(held,),
        held=held,
        rules=rules,
        published=True,
        conflicts=_basis_conflicts(source) if source is not None else (),
    )


def _published_inclusive(
    row: PublishedProviderScope,
    source: InclusiveSource | None,
    bilateral: BaseIdentity,
    facts: ScopeFacts,
    *,
    held: bool,
) -> Option:
    """The inclusive option of the published total `row` whose identity without the
    inclusive source is `bilateral` (`_published_table`)."""
    if source is not None:
        return _inclusive_option(bilateral, source, facts, held=held)
    if row.scope in _COUNTS_CONTRIBUTIONS:
        return _option("inclusive", _unavailable("contributions_counted", {"dataset": row.dataset}))
    return _option("inclusive", _unavailable("provider_not_offered", {"why": "dataset"}))


def _sibling_option(
    scope: ProviderScope,
    binding: SlotBinding,
    own: Sequence[frozenset[int]],
    held: Sequence[frozenset[int]],
) -> Option:
    """The option of `scope` binding the sibling total `binding` names, on a part asking
    for a range or every year. `own` and `held` name, for each donor of the part in order,
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
            _option(scope, binding=binding)
            if all(held)
            else _option(scope, _unavailable("option_not_covered"), binding=binding)
        )
    covered = frozenset(
        year
        for year in asked
        if all(year in theirs for ours, theirs in zip(own, held, strict=True) if year in ours)
    )
    if covered >= asked:
        return _option(scope, binding=binding)
    if not covered:
        return _option(scope, _unavailable("option_not_covered"), binding=binding)
    runs = spans(covered)
    return _option(
        scope,
        OptionAvailability(status="partial", years=runs, code="option_not_covered"),
        binding=binding.model_copy(update={"window": runs[-1]}),
        window=runs[-1],
    )


def _latest_sibling_option(
    scope: ProviderScope,
    binding: SlotBinding,
    own: Sequence[frozenset[int]],
    held: Sequence[frozenset[int]],
    counts: Mapping[int, int],
    own_year: int | None,
) -> Option:
    """The option of `scope` binding the sibling total `binding` names, on a part asking
    for the latest year, whose read elects `own_year` for the pool. `own` and `held` are as
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
        return _option(scope, _unavailable("option_not_covered"), binding=binding)
    window = (year, year)
    pinned = binding.model_copy(update={"window": window})
    if year == own_year:
        return _option(scope, binding=pinned)
    return _option(
        scope,
        OptionAvailability(status="partial", years=[window], code="option_not_covered"),
        binding=pinned,
        window=window,
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
        document = table_identity(facts.index, identity.model_dump(mode="json", exclude_none=True))
    except CompositeNotServable:
        return None
    dataset = _dataset(BaseIdentity.model_validate(document), facts)
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
    identity: BaseIdentity, entries: Sequence[Entry] | None, *, index: CodeIndex
) -> tuple[BaseIdentity, Sequence[Entry] | None]:
    """A posted `identity` and its scope `entries`, with the inclusive scope its identity
    binds read as a carried entry where its providers cannot take it.

    The inclusive composite adds the imputed share of multilateral spending to what the
    named donors' bilateral line counts. A part naming no provider, or a provider that is no
    donor alone (`_names_donors_alone`), such as "Official Donors, Total", already counts the
    multilaterals' outflows, so the composite would count them twice. Such an identity is
    served its bilateral line (`_without_inclusive`), and `provider:inclusive` joins its
    entries as a carried entry from the posted identity's provider choice, unless they hold
    a provider entry, so the scope drops it as it drops any carried entry the part cannot
    take (`compile`), and says so. `identity` and `entries` themselves otherwise."""
    try:
        ref = composite_ref(identity.model_dump(mode="json", include={"facets"}))
    except CompositeNotServable:
        return identity, entries
    if ref not in _SOURCE_COMPOSITES or _names_donors_alone(identity, index):
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
    return _without_inclusive(identity, ref), held


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


def _dataset(identity: BaseIdentity, facts: ScopeFacts) -> str | None:
    """The dataset the index elects for `identity`'s family, None when it decides none."""
    index = facts.index
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
    `bilateral` binds, even when the part holds it. For a part asking for the latest year it
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
        prop = _slot_at(bilateral, exc.pointer)
        why = _SLOT_WHY.get(prop, prop)
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
    """`identity`, served as it reads: a provider-inclusive composite asking for the latest
    year is read over the year `inclusive_reach` elects among those both its lines cover, a
    one-year range, since its lines can elect different latest years and the Datasets mint
    refuses a composite's `latest` window. Any other identity, or one whose tables elect no year, is
    `identity` itself.

    A part's base keeps its `latest` window, so the bilateral option, which removes the
    composite, compiles to the bilateral answer's own latest year; the inclusive compiles
    here to the one year both cover."""
    if identity.requested_window.kind != "latest":
        return identity
    try:
        ref = composite_ref(identity.model_dump(mode="json", include={"facets"}))
    except CompositeNotServable:
        return identity
    if ref not in _SOURCE_COMPOSITES:
        return identity
    try:
        year = inclusive_reach(_without_inclusive(identity, ref), facts, composite=ref).latest
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
    when it can: the identity pins a provenance (`provenance_pinned`), is counted on a basis
    the source does not add its imputed line on (`basis`, `InclusiveSource.bases`), or at
    constant prices, which the imputed lines are not published at (`constant_prices`)."""
    if (identity.facets or {}).get("provenance") is not None:
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


def _slot_at(identity: BaseIdentity, pointer: str) -> str:
    """The slot property at `pointer` (`/slots/<n>`) of `identity`, or the pointer itself."""
    head, _, position = pointer.rpartition("/")
    if head == "/slots" and position.isdigit() and int(position) < len(identity.slots):
        return identity.slots[int(position)].property
    return pointer


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
    dataset = _dataset(identity, facts)
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
    bilateral = _without_inclusive(identity, ref)
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
