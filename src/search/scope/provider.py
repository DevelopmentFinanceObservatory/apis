"""The provider axis: whose spending a part counts, the donor's bilateral flows alone, with
its imputed share of multilateral spending, or with its core contributions to multilaterals.

The inclusive scope binds the stacked composite of the CRS and ONE's imputed multilateral
table (`edits.INCLUSIVE_PROVENANCE`) as the identity's source; the bilateral scope removes it.
The axis is present on a part the CRS serves and on a donor's published total
(`provider_rule`). A published total holds the scope its declared row names
(`code_index.scope_conventions.PUBLISHED_PROVIDER_SCOPES`): DAC1's total ODA the
contributions scope, and DAC1's bilateral ODA the bilateral scope. Each offers the other's
scope as an option binding the other's aid type, judged per donor against the years each
donor the part names holds its own total in (`_published_table`). On a part the CRS serves,
the inclusive option serves the years both tables cover for the part (`inclusive_reach`):
available when they cover every year the bilateral answer holds, partial with a shorter
window of its own when they cover fewer, and, for a part asking for the latest year,
available over the year a `latest` read elects among those both cover. It is not offered
(`provider_not_offered`) where the imputed table cannot serve what the part binds, or
cannot count the providers it names (`_providers_not_offered`): a part naming no provider,
a provider the table carries no rows for, or a pool naming a multilateral whose outflows
the table attributes to its funders.

The default is declared (`code_index.scope_conventions.PROVIDER_CONVENTIONS`): a named donor's
total of a CRS sector or recipient part counts both, when they cover every year asked for.
`settle_provider` applies it, or the part's explicit or carried entry, on the parsed door,
where the answer can serve it (`_served_in_pass`). A choice names the convention's rule only
where the request matched the convention and the part holds its scope; otherwise it names
the rule of the option it holds, its own scope's (`AxisTable.rules`), as every resubmit
does.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import get_args

from code_index.composites import (
    DECLARED_COMPOSITES,
    CompositeNotServable,
    component_identities,
    composite_ref,
    table_identity,
    without_composite,
)
from code_index.members import SourceNotInFamily, identity_family_query, index_family
from code_index.query import CodeIndex
from code_index.scope_conventions import (
    DONOR_ORGANISATIONS,
    OPTION_LABELS,
    PROVIDER_CONVENTIONS,
    PROVIDER_SCOPE_RULES,
    PUBLISHED_PROVIDER_SCOPES,
    REDISTRIBUTED_PROVIDERS,
    ProviderConvention,
    PublishedProviderScope,
)
from materialise.composite import spans
from search.catalogue.counts import COUNTRY_PREFIX
from search.models import (
    BaseIdentity,
    IdentityBindingValue,
    IdentityFacetValueRequest,
    IdentityShapeSchema,
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
from search.scope.edits import INCLUSIVE_PROVENANCE, applies_in_pass, apply_binding
from search.scope.region import policy_members
from search.scope.state import AxisRecord, ScopeState
from search.scope.tables import AxisTable, ScopeFacts
from search.scope.vocabulary import provider_option_id
from series_coverage.index import elected_year

BILATERAL = provider_option_id("bilateral")
"""The bilateral provider option."""
INCLUSIVE = provider_option_id("inclusive")
"""The inclusive provider option."""
_SCOPE_LABELS: Mapping[ProviderScope, str] = {
    "bilateral": OPTION_LABELS.provider_bilateral,
    "inclusive": OPTION_LABELS.provider_inclusive,
    "contributions": OPTION_LABELS.provider_contributions,
}

_STACKED = DECLARED_COMPOSITES[INCLUSIVE_PROVENANCE]
_BILATERAL_TABLE = next(r.dataset for r in _STACKED.components if r.role == "bilateral")
_REDISTRIBUTED = frozenset(row.provider for row in REDISTRIBUTED_PROVIDERS)
_SCOPE_RULES: Mapping[ProviderScope, str] = {row.scope: row.rule for row in PROVIDER_SCOPE_RULES}
"""The rule each provider scope names where no convention or published row settles it."""
_DONOR_ORGANISATIONS = frozenset(DONOR_ORGANISATIONS)
_GROSS = "gross"
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


def inclusive_reach(identity: BaseIdentity, facts: ScopeFacts) -> InclusiveReach:
    """The years the inclusive scope serves of a part whose bilateral identity is
    `identity` (`InclusiveReach`), read from the coverage snapshot in memory. Raises
    `code_index.composites.CompositeNotServable` where the imputed table cannot serve what
    `identity` binds."""
    document = identity.model_dump(mode="json", exclude_none=True)
    bilateral, imputed = served_components(
        {**document, "facets": {**document.get("facets", {}), "provenance": _inclusive()}},
        facts=facts,
    )
    covered = bilateral.served
    if document["requested_window"]["kind"] == "latest":
        counts = {
            year: bilateral.years.counts[year] + imputed.years.counts[year] for year in covered
        }
        return InclusiveReach(
            asked=bilateral.years.held, covered=covered, latest=elected_year(counts)
        )
    return InclusiveReach(asked=bilateral.years.held, covered=covered)


def _inclusive() -> dict[str, str]:
    return {"state": "value", "ref": INCLUSIVE_PROVENANCE}


def provider_rule(
    base: BaseIdentity, record: AxisRecord | None, facts: ScopeFacts
) -> AxisTable | None:
    """The provider table of a part whose identity is `base`, None where the axis does not
    apply to it: `base` counts no published total (`_published_row`), and binds another
    composite than the inclusive one or, without the inclusive composite, is an identity the
    CRS does not serve.

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
    if ref is not None and ref != INCLUSIVE_PROVENANCE:
        return None
    bilateral = _without_inclusive(base, ref)
    if _dataset(bilateral, facts) != _BILATERAL_TABLE:
        return None
    held = BILATERAL if ref is None else INCLUSIVE
    options = (_option("bilateral"), _inclusive_option(bilateral, facts, held=held == INCLUSIVE))
    rules = {BILATERAL: _SCOPE_RULES["bilateral"], INCLUSIVE: _SCOPE_RULES["inclusive"]}
    return AxisTable(
        axis="provider",
        options=options,
        preference=(held, BILATERAL if held == INCLUSIVE else INCLUSIVE),
        held=held,
        rules=rules,
    )


def _without_inclusive(identity: BaseIdentity, ref: str | None) -> BaseIdentity:
    """`identity`, whose composite is `ref`, without the inclusive composite: the identity
    the CRS serves its bilateral flows on. `identity` itself when `ref` is another one."""
    if ref != INCLUSIVE_PROVENANCE:
        return identity
    return BaseIdentity.model_validate(
        without_composite(identity.model_dump(mode="json", exclude_none=True))
    )


def _published_table(
    row: PublishedProviderScope, identity: BaseIdentity, *, facts: ScopeFacts
) -> AxisTable:
    """The provider table of a part counting the published total `row` declares, whose
    identity is `identity` (`_published_row`).

    The part holds `row`'s scope. Every other row of its dataset and code system
    (`PUBLISHED_PROVIDER_SCOPES`) is an option binding that row's code, so a switch to it
    reads the total that counts its scope: on DAC1's total ODA, bilateral flows bind
    bilateral ODA, and on bilateral ODA, core contributions bind total ODA. Such an option
    serves the years each donor of the part holds its total in (`_sibling_option`), or, on a
    part asking for the latest year, the one year the pool's read elects
    (`_latest_sibling_option`). The inclusive scope is unavailable with the total's dataset:
    `contributions_counted` on a total counting core contributions, since it counts them in
    place of an imputed share of the multilaterals' spending, and `provider_not_offered` (why
    `dataset`) on any other, since the imputed table stacks on the CRS alone. Each option
    names its row's rule. No preference but the held scope leads, so no default moves the
    part to another total."""
    rows = {
        sibling.scope: sibling
        for sibling in PUBLISHED_PROVIDER_SCOPES
        if (sibling.dataset, sibling.code_system) == (row.dataset, row.code_system)
    }
    options: list[Option] = []
    rules: dict[str, str] = {}
    own = _donor_years(identity, facts)
    own_year = (
        elected_year(_year_counts(identity, facts))
        if identity.requested_window.kind == "latest"
        else None
    )
    for scope in get_args(ProviderScope):
        sibling = rows.get(scope)
        if sibling is None and scope != "inclusive":
            continue
        option_id = provider_option_id(scope)
        rules[option_id] = sibling.rule if sibling is not None else _SCOPE_RULES[scope]
        if sibling is None:
            counted = row.scope == "contributions"
            options.append(
                _option(
                    scope,
                    _unavailable(
                        "contributions_counted" if counted else "provider_not_offered",
                        {"dataset": row.dataset} if counted else {"why": "dataset"},
                    ),
                )
            )
            continue
        if sibling is row:
            options.append(_option(scope, binding=ProviderBinding(scope=scope)))
            continue
        binding = SlotBinding(slot=sibling.code_system, state="value", refs=[sibling.code])
        switched = apply_binding(identity, binding, [])
        sibling_years = _donor_years(switched, facts)
        options.append(
            _sibling_option(scope, binding, own, sibling_years)
            if identity.requested_window.kind != "latest"
            else _latest_sibling_option(
                scope, binding, own, sibling_years, _year_counts(switched, facts), own_year
            )
        )
    held = provider_option_id(row.scope)
    return AxisTable(
        axis="provider",
        options=tuple(options),
        preference=(held,),
        held=held,
        rules=rules,
        published=True,
    )


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
    label states, and unavailable when none is. Each reason is `option_not_covered`."""
    asked = frozenset[int]().union(*own)
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
    unavailable when the donors share no year. Each reason is `option_not_covered`."""
    demanded = [
        theirs
        for ours, theirs in zip(own, held, strict=True)
        if own_year is None or own_year in ours
    ]
    shared = frozenset[int].intersection(*demanded)
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
) -> Option:
    """The option of `scope`, with `availability`, available when None, binding `binding` or
    else the scope itself over `window`. A `window` also states its years in the label
    (`OPTION_LABELS.provider_years`): the years both tables of the inclusive scope cover, or
    the run a partial sibling total serves."""
    label = (
        _SCOPE_LABELS[scope]
        if window is None
        else OPTION_LABELS.provider_years.format(
            label=_SCOPE_LABELS[scope], years=_years_label(window)
        )
    )
    return Option(
        option_id=provider_option_id(scope),
        label=label,
        binding=binding if binding is not None else ProviderBinding(scope=scope, window=window),
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
    institutions and a donor outside the EU counts no contribution twice."""
    bound = {
        (slot.property, slot.binding.ref)
        for slot in identity.slots
        if isinstance(slot.binding, IdentityBindingValue)
    }
    rows = [row for row in PUBLISHED_PROVIDER_SCOPES if (row.code_system, row.code) in bound]
    providers = identity.entities.get("origin") or []
    if not rows or not providers:
        return None
    if _pools_funder(providers, facts.index) or not all(
        _counts_as_donor(provider, facts.index) for provider in providers
    ):
        return None
    try:
        document = table_identity(facts.index, identity.model_dump(mode="json", exclude_none=True))
    except CompositeNotServable:
        return None
    dataset = _dataset(BaseIdentity.model_validate(document), facts)
    return next((row for row in rows if row.dataset == dataset), None)


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


def _pools_funder(providers: Collection[str], index: CodeIndex) -> bool:
    """Whether `providers` pool a redistributed provider (`REDISTRIBUTED_PROVIDERS`) with a
    member of the group that funds it, or a total that sums one (`_members`): the EU
    institutions with an EU member state, or with the G7. A published total counts such a
    pool's core contributions to the provider twice. Where the policy holds no members for
    the funding group, any pool with the provider counts as one."""
    for row in REDISTRIBUTED_PROVIDERS:
        if row.provider not in providers:
            continue
        funders = policy_members(row.funders)
        others = (provider for provider in providers if provider != row.provider)
        if any(not funders or funders & _members(other, index) for other in others):
            return True
    return False


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


def _inclusive_option(bilateral: BaseIdentity, facts: ScopeFacts, *, held: bool) -> Option:
    """The inclusive option over `bilateral`, the part's identity without the composite.
    It is not offered for what `_not_offered` and `_providers_not_offered` refuse, or where
    the imputed table cannot serve a slot `bilateral` binds, even when the part holds it.
    For a part asking for the latest year it is available over the year `inclusive_reach`
    elects, which its window states (`elect_latest_window`). Otherwise it is available when
    both tables cover every year the bilateral answer holds; partial over the latest run of
    years they cover, with that run as its window, when they cover fewer; and unavailable
    when they cover none. The option a part holds over a range or every year is available
    over its own window."""
    try:
        why = _not_offered(bilateral) or _providers_not_offered(bilateral, facts)
    except CompositeNotServable as exc:
        prop = _slot_at(bilateral, exc.pointer)
        why = _SLOT_WHY.get(prop, prop)
    if why is not None:
        return _option("inclusive", _unavailable("provider_not_offered", {"why": why}))
    latest = bilateral.requested_window.kind == "latest"
    if held and not latest:
        return _option("inclusive")
    reach = inclusive_reach(bilateral, facts)
    if latest:
        if reach.latest is None:
            return _option("inclusive", _unavailable("imputed_coverage"))
        return _option("inclusive", window=(reach.latest, reach.latest))
    if not reach.covered:
        return _option("inclusive", _unavailable("imputed_coverage"))
    if reach.covered >= reach.asked:
        return _option("inclusive")
    runs = spans(reach.covered)
    return _option(
        "inclusive",
        OptionAvailability(status="partial", years=runs, code="imputed_coverage"),
        window=runs[-1],
    )


def elect_latest_window(identity: BaseIdentity, facts: ScopeFacts) -> BaseIdentity:
    """`identity`, served as it reads: a provider-inclusive composite asking for the latest
    year is read over the year `inclusive_reach` elects among those both its tables cover, a
    one-year range, since its tables can elect different latest years and the Datasets mint
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
    if ref != INCLUSIVE_PROVENANCE:
        return identity
    try:
        year = inclusive_reach(_without_inclusive(identity, ref), facts).latest
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


def _not_offered(identity: BaseIdentity) -> str | None:
    """Why the inclusive scope cannot serve `identity`, None when it can: the identity pins
    a provenance (`provenance_pinned`), is counted on a basis other than gross disbursements,
    the imputed table's only one (`basis`), or at constant prices, which the table does not
    publish (`constant_prices`)."""
    if (identity.facets or {}).get("provenance") is not None:
        return "provenance_pinned"
    measure = held_measure(identity)
    if measure is None or measure.basis != _GROSS:
        return "basis"
    shape = identity.shape
    unit = (identity.facets or {}).get("unit")
    if (isinstance(shape, IdentityShapeSchema) and shape.measurement_qualifier == _CONSTANT) or (
        isinstance(unit, IdentityFacetValueRequest) and unit.request.prices != "current"
    ):
        return "constant_prices"
    return None


def _providers_not_offered(identity: BaseIdentity, facts: ScopeFacts) -> str | None:
    """Why the inclusive scope cannot count the providers `identity` names, None when it can.

    The imputed table attributes each multilateral's spending to the providers that fund it,
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
    components = component_identities(
        facts.index,
        {**document, "facets": {**document.get("facets", {}), "provenance": _inclusive()}},
    )
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
    option the pass can serve (`_served_in_pass`) settles it. A partial option is not
    applied from words or a carried entry, since only a switch accepts fewer years than
    asked for. Otherwise the declared convention matching the part (`provider_convention`,
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
        if _served_in_pass(table, entry.option_id):
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


def _served_in_pass(table: AxisTable, option_id: str) -> bool:
    """Whether a fresh search serves option `option_id` of `table` as its answer: it is
    available, and it is the option the answer holds or one binding a composite, which the
    scope applies to the answer (`edits.applies_in_pass`, `build._with_recorded_composites`)."""
    option = table.option(option_id)
    if option is None or option.availability.status != "available":
        return False
    return option_id == table.held or applies_in_pass(option.binding)
