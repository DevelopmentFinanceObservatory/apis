"""The provider axis: whose spending a part counts, the donor's bilateral flows alone, with
its imputed share of multilateral spending, or with its core contributions to multilaterals.

The inclusive scope binds the stacked composite of the CRS and ONE's imputed multilateral
table (`edits.INCLUSIVE_PROVENANCE`) as the identity's source; the bilateral scope removes it.
The axis is present on a part the CRS serves and on a donor's published total
(`provider_rule`). A published total holds the scope its declared row names
(`code_index.scope_conventions.PUBLISHED_PROVIDER_SCOPES`): DAC1's total ODA the
contributions scope, and DAC1's bilateral ODA the bilateral scope. Each offers the other's
scope as an option binding the other's aid type (`_published_table`). On a part the CRS
serves, the inclusive option serves the years both tables cover for the part
(`inclusive_reach`): available when they cover every year the bilateral answer holds,
partial with a shorter window of its own when they cover fewer, and, for a part asking for
the latest year, available over the year a `latest` read elects among those both cover. It
is not offered (`provider_not_offered`) where the imputed table cannot serve what the part
binds, or cannot count the providers it names (`_providers_not_offered`): a part naming no
provider, a provider the table carries no rows for, or a pool naming a multilateral whose
outflows the table attributes to its funders.

The default is declared (`code_index.scope_conventions.PROVIDER_CONVENTIONS`): a named donor's
total of a CRS sector or recipient part counts both, when they cover every year asked for.
`settle_provider` applies it, or the part's explicit or carried entry, on the parsed door,
where the answer can serve it (`_served_in_pass`). The rule a part's choice names is the one
the option it holds names (`AxisTable.rules`), on a fresh search and a resubmit alike.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
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
    PROVIDER_DEFAULT_RULE,
    PUBLISHED_PROVIDER_SCOPES,
    REDISTRIBUTED_PROVIDERS,
    ProviderConvention,
    PublishedProviderScope,
)
from materialise.composite import spans
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
from search.scope.composite import component_years, served_components
from search.scope.edits import INCLUSIVE_PROVENANCE
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
_REDISTRIBUTED = frozenset(REDISTRIBUTED_PROVIDERS)
_DONOR_ORGANISATIONS = frozenset(DONOR_ORGANISATIONS)
_COUNTRY = "country/"
"""The dcid namespace of the graph's countries."""
_AVAILABLE = OptionAvailability(status="available")
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

    Read from `base` alone, so a resubmit of a served identity offers what its search
    offered. A published total offers what `_published_table` states. On a part the CRS
    serves, the option `base` holds leads the preference, so a part's default is the scope it
    resolved; the bilateral option is always available, and the inclusive option is
    available, partial or unavailable by `inclusive_reach`, or not offered (`_not_offered`).
    Each option names its own rule when the part holds it (`AxisTable.rules`), so a fresh
    search's record (`settle_provider`), its choice and a resubmit's choice all name the rule
    of the option the part holds. The table is read once per scope build for `base`, so
    `record`, whose option the table offers and whose rule the table names, adds nothing."""
    del record
    key = "provider_table:" + base.model_dump_json(exclude_none=True)
    return facts.remembered(key, lambda: _table(base, facts))


def _table(base: BaseIdentity, facts: ScopeFacts) -> AxisTable | None:
    """The provider table of `base` (`provider_rule`), None where the axis does not apply."""
    published = _published_row(base, facts)
    if published is not None:
        return _published_table(published)
    try:
        ref = composite_ref(base.model_dump(mode="json", include={"facets"}))
    except CompositeNotServable:
        return None
    if ref is not None and ref != INCLUSIVE_PROVENANCE:
        return None
    bilateral = (
        base
        if ref is None
        else BaseIdentity.model_validate(
            without_composite(base.model_dump(mode="json", exclude_none=True))
        )
    )
    if _dataset(bilateral, facts) != _BILATERAL_TABLE:
        return None
    held = BILATERAL if ref is None else INCLUSIVE
    options = (_option("bilateral"), _inclusive_option(bilateral, facts, held=held == INCLUSIVE))
    convention = provider_convention(bilateral, facts, asks_total=True)
    convention_option = provider_option_id(convention.scope) if convention is not None else None
    rules = {
        option.option_id: (
            convention.rule
            if convention is not None and option.option_id == convention_option
            else PROVIDER_DEFAULT_RULE
        )
        for option in options
    }
    return AxisTable(
        axis="provider",
        rule=rules[held],
        options=options,
        preference=(held, BILATERAL if held == INCLUSIVE else INCLUSIVE),
        held=held,
        rules=rules,
    )


def _published_table(row: PublishedProviderScope) -> AxisTable:
    """The provider table of a part counting the published total `row` declares.

    The part holds `row`'s scope. Every other row of its dataset and code system
    (`PUBLISHED_PROVIDER_SCOPES`) is an option binding that row's code, so a switch to it
    reads the total that counts its scope: on DAC1's total ODA, bilateral flows bind
    bilateral ODA, and on bilateral ODA, core contributions bind total ODA. The inclusive scope is unavailable with the
    total's dataset: `contributions_counted` on a total counting core contributions, since it
    counts them in place of an imputed share of the multilaterals' spending, and
    `provider_not_offered` (why `dataset`) on any other, since the imputed table stacks on
    the CRS alone. Each option names its row's rule. No preference but the held scope leads,
    so no default moves the part to another total."""
    rows = {
        sibling.scope: sibling
        for sibling in PUBLISHED_PROVIDER_SCOPES
        if (sibling.dataset, sibling.code_system) == (row.dataset, row.code_system)
    }
    options: list[Option] = []
    rules: dict[str, str] = {}
    for scope in get_args(ProviderScope):
        sibling = rows.get(scope)
        if sibling is None and scope != "inclusive":
            continue
        option_id = provider_option_id(scope)
        rules[option_id] = sibling.rule if sibling is not None else PROVIDER_DEFAULT_RULE
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
        binding = (
            ProviderBinding(scope=scope)
            if sibling is row
            else SlotBinding(slot=sibling.code_system, state="value", refs=[sibling.code])
        )
        options.append(_option(scope, binding=binding))
    held = provider_option_id(row.scope)
    return AxisTable(
        axis="provider",
        rule=rules[held],
        options=tuple(options),
        preference=(held,),
        held=held,
        rules=rules,
    )


def _option(
    scope: ProviderScope,
    availability: OptionAvailability = _AVAILABLE,
    *,
    binding: OptionBinding | None = None,
    window: YearSpan | None = None,
) -> Option:
    """The option of `scope`, with `availability`, binding `binding` or else the scope itself
    over `window` when it sets one, as the inclusive scope over the years both its tables
    cover (`OPTION_LABELS.provider_inclusive_years`)."""
    label = (
        _SCOPE_LABELS[scope]
        if window is None
        else OPTION_LABELS.provider_inclusive_years.format(years=_years_label(window))
    )
    return Option(
        option_id=provider_option_id(scope),
        label=label,
        binding=binding if binding is not None else ProviderBinding(scope=scope, window=window),
        availability=availability,
    )


def _unavailable(
    code: ScopeReasonCode, detail: Mapping[str, str] | None = None
) -> OptionAvailability:
    """An unavailable option's availability, for `code` with `detail`."""
    return OptionAvailability(status="unavailable", code=code, detail=dict(detail or {}))


def _published_row(identity: BaseIdentity, facts: ScopeFacts) -> PublishedProviderScope | None:
    """The declared row (`PUBLISHED_PROVIDER_SCOPES`) of the total `identity` counts: one
    whose code `identity` binds as a value and whose dataset the index elects for the table
    identity `identity` reads (`table_identity`, the net identity of the OECD headline), when
    every provider `identity` names counts as a donor (`_counts_as_donor`) and they do not
    pool a redistributed provider (`_pools_redistributed`). None otherwise, or when the
    composite `identity` binds cannot derive its components. The bound codes and the
    providers are read before the family lookup.

    A pool of an EU member and the EU institutions counts the member's core contributions to
    the institutions twice in total ODA, as the member's contributions and as the
    institutions' own aid, and a switch from bilateral ODA would reach that total, so such a
    pool has no record: every scope a record would hold or offer for it is either one it
    double counts or one a switch leaves for one it double counts."""
    bound = {
        (slot.property, slot.binding.ref)
        for slot in identity.slots
        if isinstance(slot.binding, IdentityBindingValue)
    }
    rows = [row for row in PUBLISHED_PROVIDER_SCOPES if (row.code_system, row.code) in bound]
    providers = identity.entities.get("origin") or []
    if not rows or not providers:
        return None
    if _pools_redistributed(providers) or not all(
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
    `DONOR_ORGANISATIONS` declares, or a total (`PartyRow.totals`) of a group whose
    membership (`region.policy_members`) holds countries alone, as the G7 or the DAC
    countries. Any other organisation or total, the DAC's total of the multilaterals among
    them, counts multilateral outflows."""
    if party.startswith(_COUNTRY) or party in _DONOR_ORGANISATIONS:
        return True
    row = index.parties.get(party)
    if row is None or row.totals is None:
        return False
    members = policy_members(row.totals)
    return bool(members) and all(member.startswith(_COUNTRY) for member in members)


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
    bilateral = BaseIdentity.model_validate(
        without_composite(identity.model_dump(mode="json", exclude_none=True))
    )
    try:
        year = inclusive_reach(bilateral, facts).latest
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
    does."""
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
    applied from words or a carried entry, since only a switch shortens the years asked for.
    Otherwise the declared convention matching the part (`provider_convention`, with the
    request's total intent) settles its scope where the pass can serve it, else the
    bilateral scope, else the scope `identity` holds. A published total matches no
    convention, so it holds the scope its row declares. The record names the rule of the
    option it settles (`AxisTable.rule_of`), which a resubmit of the identity the scope
    compiles to names too."""
    table = provider_rule(identity, None, facts)
    if table is None:
        return None
    convention = provider_convention(identity, facts, asks_total=state.total)
    wanted = provider_option_id(convention.scope) if convention is not None else table.held
    return _settled(table, state, wanted=wanted)


def _settled(table: AxisTable, state: ScopeState, *, wanted: str) -> AxisRecord:
    """The record of `table`: the first option an entry on the axis asks for that the pass
    serves, else the first of `wanted` and the bilateral option that it serves, else the
    option the part holds, under the rule that option names."""
    for entry in state.entries_on("provider"):
        if _served_in_pass(table, entry.option_id):
            return AxisRecord(
                axis="provider",
                rule=table.rule_of(entry.option_id),
                active=entry.option_id,
                provenance=entry.provenance,
                entry=entry,
            )
    active = next((o for o in (wanted, BILATERAL) if _served_in_pass(table, o)), table.held)
    return AxisRecord(
        axis="provider", rule=table.rule_of(active), active=active, provenance="default"
    )


def _served_in_pass(table: AxisTable, option_id: str) -> bool:
    """Whether a fresh search serves option `option_id` of `table` as its answer: it is
    available, and it is the option the answer holds or the inclusive scope, whose composite
    the scope applies to the answer (`build._with_recorded_composites`). Any other option
    reads another identity, which a switch or resubmit serves."""
    option = table.option(option_id)
    if option is None or option.availability.status != "available":
        return False
    binding = option.binding
    return option_id == table.held or (
        isinstance(binding, ProviderBinding) and binding.scope == "inclusive"
    )
