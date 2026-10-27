"""The provider axis: whose spending a part counts, the donor's bilateral flows alone or with
its imputed share of multilateral spending.

The inclusive scope binds the stacked composite of the CRS and ONE's imputed multilateral
table (`edits.INCLUSIVE_PROVENANCE`) as the identity's source; the bilateral scope removes it.
The axis is present on a part the CRS serves (`provider_rule`). The inclusive option serves the
years both tables cover for the part (`inclusive_reach`): available when they cover every year
the bilateral answer holds, partial with a shorter window of its own when they cover fewer,
and, for a part asking for the latest year, available over the latest year both cover. It is
not offered (`provider_not_offered`) where the imputed table cannot serve what the part binds.

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
    without_composite,
)
from code_index.members import SourceNotInFamily, identity_family_query, index_family
from code_index.scope_conventions import (
    OPTION_LABELS,
    PROVIDER_CONVENTIONS,
    PROVIDER_DEFAULT_RULE,
    ProviderConvention,
)
from search.models import (
    BaseIdentity,
    IdentityFacetValueRequest,
    IdentityShapeSchema,
    IdentityWindowRange,
    Option,
    OptionAvailability,
    ProviderBinding,
    YearSpan,
)
from search.scope.basis import held_measure, identity_part_shape
from search.scope.composite import component_years, spans
from search.scope.edits import INCLUSIVE_PROVENANCE
from search.scope.state import AxisRecord, ScopeState
from search.scope.tables import AxisTable, ScopeFacts
from search.scope.vocabulary import provider_option_id

BILATERAL = provider_option_id("bilateral")
"""The bilateral provider option."""
INCLUSIVE = provider_option_id("inclusive")
"""The inclusive provider option."""

_STACKED = DECLARED_COMPOSITES[INCLUSIVE_PROVENANCE]
_BILATERAL_TABLE = next(r.dataset for r in _STACKED.components if r.role == "bilateral")
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
    holds in the window it asks for, and `covered`, those the imputed table holds too. For a
    part asking for the latest year, `latest` is the latest year both tables cover, the
    earlier of the years each elects; None otherwise, or when one elects none."""

    asked: frozenset[int]
    covered: frozenset[int]
    latest: int | None = None


def inclusive_reach(identity: BaseIdentity, facts: ScopeFacts) -> InclusiveReach:
    """The years the inclusive scope serves of a part whose bilateral identity is
    `identity` (`InclusiveReach`), read from the coverage snapshot in memory. Raises
    `code_index.composites.CompositeNotServable` where the imputed table cannot serve what
    `identity` binds."""
    document = identity.model_dump(mode="json", exclude_none=True)
    components = component_identities(
        facts.index,
        {**document, "facets": {**document.get("facets", {}), "provenance": _inclusive()}},
    )
    assert components is not None
    bilateral, imputed = (
        component_years(component.identity, facts=facts) for component in components
    )
    if document["requested_window"]["kind"] == "latest":
        elected = (bilateral.latest, imputed.latest)
        latest = None if None in elected else min(year for year in elected if year is not None)
        return InclusiveReach(asked=bilateral.held, covered=imputed.held, latest=latest)
    return InclusiveReach(asked=bilateral.held, covered=bilateral.held & imputed.held)


def _inclusive() -> dict[str, str]:
    return {"state": "value", "ref": INCLUSIVE_PROVENANCE}


def provider_rule(
    base: BaseIdentity, record: AxisRecord | None, facts: ScopeFacts
) -> AxisTable | None:
    """The provider table of a part whose identity is `base`, None where the axis does not
    apply to it: `base` binds another composite, or the CRS does not serve its bilateral
    identity (`base` without the inclusive composite).

    Read from `base` alone, so a resubmit of a served identity offers what its search
    offered. The option `base` holds is available and leads the preference, so a part's
    default is the scope it resolved. The bilateral option is always available. The inclusive
    option is available, partial or unavailable by `inclusive_reach`, or not offered
    (`_not_offered`). The rule is the record's when a binder wrote one, else `_held_rule`'s."""
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
    For a part asking for the latest year it is available over the latest year both tables
    cover, which its window states (`elect_latest_window`). Otherwise it is available when
    both tables cover every year the bilateral answer holds; partial over the latest run of
    years they cover, with that run as its window, when they cover fewer; and unavailable
    when they cover none. The option a part holds over a range or every year is available
    over its own window."""
    why = _not_offered(bilateral)
    if why is not None:
        return _option(
            OptionAvailability(
                status="unavailable", code="provider_not_offered", detail={"why": why}
            )
        )
    latest = bilateral.requested_window.kind == "latest"
    if held and not latest:
        return _option(OptionAvailability(status="available"))
    try:
        reach = inclusive_reach(bilateral, facts)
    except CompositeNotServable as exc:
        prop = _slot_at(bilateral, exc.pointer)
        return _option(
            OptionAvailability(
                status="unavailable",
                code="provider_not_offered",
                detail={"why": _SLOT_WHY.get(prop, prop)},
            )
        )
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
    year is read over the latest year both its tables cover (`inclusive_reach`), a one-year
    range, since its tables can elect different latest years and the Datasets mint refuses a
    composite's `latest` window. Any other identity, or one whose tables elect no year, is
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
    settles its scope when the scope is available, and the bilateral scope stands otherwise.
    The record's rule is the matching convention's, or `PROVIDER_DEFAULT_RULE`."""
    table = provider_rule(identity, None, facts)
    if table is None:
        return None
    convention = provider_convention(identity, facts, asks_total=state.total)
    rule = convention.rule if convention is not None else PROVIDER_DEFAULT_RULE
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
    wanted = provider_option_id(convention.scope) if convention is not None else BILATERAL
    option = table.option(wanted)
    active = (
        wanted if option is not None and option.availability.status == "available" else BILATERAL
    )
    return AxisRecord(axis="provider", rule=rule, active=active, provenance="default")
