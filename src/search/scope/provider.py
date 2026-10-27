"""The provider axis: whose spending a part counts, the donor's bilateral flows alone, with
its imputed share of multilateral spending, or with its core contributions to multilaterals.

The inclusive scope binds the stacked composite of the CRS and ONE's imputed multilateral
table (`edits.INCLUSIVE_PROVENANCE`) as the identity's source; the bilateral scope removes it.
The axis is present on a part the CRS serves and on a published donor total that counts core
contributions (`provider_rule`). Such a total, DAC1's total ODA
(`code_index.scope_conventions.CONTRIBUTION_TOTALS`), offers the contributions scope it
publishes as its one available option, and the bilateral and inclusive scopes unavailable
(`counts_core_contributions`), so a calculation comparing provider scopes never reads it as
either. On a part the CRS serves, the inclusive option serves the
years both tables cover for the part (`inclusive_reach`): available when they cover every year
the bilateral answer holds, partial with a shorter window of its own when they cover fewer,
and, for a part asking for the latest year, available over the year a `latest` read elects
among those both cover. It is not offered (`provider_not_offered`) where the imputed table
cannot serve what the part binds, or cannot count the providers it names
(`_providers_not_offered`): a part naming no provider, a provider the table carries no rows
for, or a pool naming a multilateral whose outflows the table attributes to its funders.

The default is declared (`code_index.scope_conventions.PROVIDER_CONVENTIONS`): a named donor's
total of a CRS sector or recipient part counts both, when they cover every year asked for.
`settle_provider` applies it, or the part's explicit or carried entry, on the parsed door.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from code_index.composites import (
    DECLARED_COMPOSITES,
    CompositeNotServable,
    component_identities,
    composite_ref,
    table_identity,
    without_composite,
)
from code_index.members import SourceNotInFamily, identity_family_query, index_family
from code_index.scope_conventions import (
    CONTRIBUTION_TOTALS,
    CONTRIBUTIONS_RULE,
    OPTION_LABELS,
    PROVIDER_CONVENTIONS,
    PROVIDER_DEFAULT_RULE,
    REDISTRIBUTED_PROVIDERS,
    ContributionTotal,
    ProviderConvention,
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
    ProviderBinding,
    YearSpan,
)
from search.scope.basis import held_measure, identity_part_shape
from search.scope.composite import component_years, served_components
from search.scope.edits import INCLUSIVE_PROVENANCE
from search.scope.state import AxisRecord, ScopeState
from search.scope.tables import AxisTable, ScopeFacts
from search.scope.vocabulary import provider_option_id
from series_coverage.index import elected_year

BILATERAL = provider_option_id("bilateral")
"""The bilateral provider option."""
INCLUSIVE = provider_option_id("inclusive")
"""The inclusive provider option."""
CONTRIBUTIONS = provider_option_id("contributions")
"""The provider option of a published donor total that counts core contributions."""

_STACKED = DECLARED_COMPOSITES[INCLUSIVE_PROVENANCE]
_BILATERAL_TABLE = next(r.dataset for r in _STACKED.components if r.role == "bilateral")
_REDISTRIBUTED = frozenset(REDISTRIBUTED_PROVIDERS)
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
    apply to it: `base` binds neither the inclusive composite over an identity the CRS
    serves nor a published total that counts core contributions (`_contributions_table`).

    Read from `base` alone, so a resubmit of a served identity offers what its search
    offered. The option `base` holds is available and leads the preference, so a part's
    default is the scope it resolved. On a part the CRS serves, the bilateral option is
    always available, and the inclusive option is available, partial or unavailable by
    `inclusive_reach`, or not offered (`_not_offered`). The rule is the record's when a
    binder wrote one, else `_held_rule`'s."""
    try:
        ref = composite_ref(base.model_dump(mode="json", include={"facets"}))
    except CompositeNotServable:
        return None
    if ref is not None and ref != INCLUSIVE_PROVENANCE:
        return _contributions_table(base, record, facts)
    bilateral = (
        base
        if ref is None
        else BaseIdentity.model_validate(
            without_composite(base.model_dump(mode="json", exclude_none=True))
        )
    )
    if _dataset(bilateral, facts) != _BILATERAL_TABLE:
        return _contributions_table(base, record, facts) if ref is None else None
    held = BILATERAL if ref is None else INCLUSIVE
    options = (
        Option(
            option_id=BILATERAL,
            label=OPTION_LABELS.provider_bilateral,
            binding=ProviderBinding(scope="bilateral"),
            availability=OptionAvailability(status="available"),
        ),
        _inclusive_option(bilateral, facts, held=held == INCLUSIVE),
    )
    return AxisTable(
        axis="provider",
        rule=record.rule if record is not None else _held_rule(bilateral, facts, held=held),
        options=options,
        preference=(held, BILATERAL if held == INCLUSIVE else INCLUSIVE),
        held=held,
    )


def _contributions_table(
    base: BaseIdentity, record: AxisRecord | None, facts: ScopeFacts
) -> AxisTable | None:
    """The provider table of a part counting a published donor total that holds core
    contributions to multilaterals (`_contribution_total`), None for any other part.

    The contributions option is the one the total holds and its only available one. The
    bilateral and inclusive options are unavailable (`counts_core_contributions`, with the
    total's dataset): the total cannot drop the contributions it counts, nor replace them
    with imputed shares of multilateral spending, and the table's own bilateral line is
    another total, which no option of this part binds. The rule is the record's when a
    binder wrote one, else `CONTRIBUTIONS_RULE`."""
    total = _contribution_total(base, facts)
    if total is None:
        return None
    published = OptionAvailability(
        status="unavailable", code="counts_core_contributions", detail={"dataset": total.dataset}
    )
    options = (
        Option(
            option_id=BILATERAL,
            label=OPTION_LABELS.provider_bilateral,
            binding=ProviderBinding(scope="bilateral"),
            availability=published,
        ),
        Option(
            option_id=INCLUSIVE,
            label=OPTION_LABELS.provider_inclusive,
            binding=ProviderBinding(scope="inclusive"),
            availability=published,
        ),
        Option(
            option_id=CONTRIBUTIONS,
            label=OPTION_LABELS.provider_contributions,
            binding=ProviderBinding(scope="contributions"),
            availability=OptionAvailability(status="available"),
        ),
    )
    return AxisTable(
        axis="provider",
        rule=record.rule if record is not None else CONTRIBUTIONS_RULE,
        options=options,
        preference=(CONTRIBUTIONS,),
        held=CONTRIBUTIONS,
    )


def _contribution_total(identity: BaseIdentity, facts: ScopeFacts) -> ContributionTotal | None:
    """The declared total (`CONTRIBUTION_TOTALS`) `identity` counts: the one whose dataset
    the index elects for the table identity `identity` reads (`table_identity`, the net
    identity of the OECD headline) and whose code that identity binds as a value. None when
    no total matches, or the composite `identity` binds cannot derive its components."""
    try:
        document = table_identity(facts.index, identity.model_dump(mode="json", exclude_none=True))
    except CompositeNotServable:
        return None
    table = BaseIdentity.model_validate(document)
    dataset = _dataset(table, facts)
    bound = {
        (slot.property, slot.binding.ref)
        for slot in table.slots
        if isinstance(slot.binding, IdentityBindingValue)
    }
    return next(
        (
            total
            for total in CONTRIBUTION_TOTALS
            if total.dataset == dataset and (total.code_system, total.code) in bound
        ),
        None,
    )


def _held_rule(bilateral: BaseIdentity, facts: ScopeFacts, *, held: str) -> str:
    """The rule of a part no binder recorded (a resubmit): the convention that settles the
    inclusive scope it holds, else `PROVIDER_DEFAULT_RULE`."""
    convention = provider_convention(bilateral, facts, asks_total=True)
    if held == INCLUSIVE and convention is not None and convention.scope == "inclusive":
        return convention.rule
    return PROVIDER_DEFAULT_RULE


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
        return _option(
            OptionAvailability(
                status="unavailable", code="provider_not_offered", detail={"why": why}
            )
        )
    latest = bilateral.requested_window.kind == "latest"
    if held and not latest:
        return _option(OptionAvailability(status="available"))
    reach = inclusive_reach(bilateral, facts)
    if latest:
        if reach.latest is None:
            return _option(OptionAvailability(status="unavailable", code="imputed_coverage"))
        return _option(OptionAvailability(status="available"), window=(reach.latest, reach.latest))
    if not reach.covered:
        return _option(OptionAvailability(status="unavailable", code="imputed_coverage"))
    if reach.covered >= reach.asked:
        return _option(OptionAvailability(status="available"))
    runs = spans(reach.covered)
    return _option(
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


def _option(availability: OptionAvailability, *, window: YearSpan | None = None) -> Option:
    """The inclusive option, with `availability` and over `window` when it sets one."""
    label = (
        OPTION_LABELS.provider_inclusive
        if window is None
        else OPTION_LABELS.provider_inclusive_years.format(years=_years_label(window))
    )
    return Option(
        option_id=INCLUSIVE,
        label=label,
        binding=ProviderBinding(scope="inclusive", window=window),
        availability=availability,
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
    return None if carried and _REDISTRIBUTED.isdisjoint(providers) else "pooled_providers"


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
    option is available settles it. A partial option is not applied from words or a
    carried entry, since only a switch shortens the years asked for. Otherwise the declared
    convention matching the part (`provider_convention`, with the request's total intent)
    settles its scope when the scope is available, and the scope `identity` holds stands
    otherwise: the bilateral scope on a part the CRS serves, the contributions scope on a
    published total that counts them. The record's rule is the matching convention's, or
    the table's (`PROVIDER_DEFAULT_RULE` on a part the CRS serves)."""
    table = provider_rule(identity, None, facts)
    if table is None:
        return None
    convention = provider_convention(identity, facts, asks_total=state.total)
    rule = convention.rule if convention is not None else table.rule
    for entry in state.entries_on("provider"):
        option = table.option(entry.option_id)
        if option is not None and option.availability.status == "available":
            return AxisRecord(
                axis="provider",
                rule=rule,
                active=entry.option_id,
                provenance=entry.provenance,
                entry=entry,
            )
    wanted = provider_option_id(convention.scope) if convention is not None else table.held
    option = table.option(wanted)
    active = (
        wanted if option is not None and option.availability.status == "available" else table.held
    )
    return AxisRecord(axis="provider", rule=rule, active=active, provenance="default")
