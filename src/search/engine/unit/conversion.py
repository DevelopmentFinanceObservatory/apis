"""The unit conversions an answer offers (`ResolvedSpec.unit_conversion`), and the years a
unit request converts (`conversion_period`).

An answer converts when its served members share one conversion recipe: every member is
in nominal prices with a population type the recipe converts, and every served
provenance names the recipe. The answer's unit is a currency, or a unit request already
converted it.

`ConversionAvailability` decides which years convert, for an answer and a target
(currencies, price basis, base year). A row converts only when the tables hold every term
`materialise.convert` reads for it (`materialise.convert.row_terms`): the native
currency's rate in the row's year, and at current prices the target currency's rate in that
year, or at constant prices the deflator of the row's entity on the recipe's deflator role
in that year and in the base year, and the target's rate in the base year. The deflator
key follows the recipe's fallback rule (`RecipeTables.deflator_years_of`). A missing rate
refuses the conversion, so the rates bound the years of every row. A missing deflator
leaves out only the rows of its entity, so the deflators bound a year only when no
entity's deflator holds it, and an entity whose deflator lacks the base year loses its
rows and bounds no base year. The offers, the period a unit request shows
(`convertibility.unit_request_period`), the rows each count keeps (`CountedConversion`) and
the conversion itself read those terms, so an offer never names a year or base year the
conversion refuses, and a count states the rows the conversion writes.

The answer's native currencies are those of the facets its read plans serve
(`read_plan.served_units`), whatever unit request the answer carries: a request served
natively, such as current US dollars on a dollar series, leaves the rows in their native
currency, and a later conversion still divides by its rates. Its deflator entities are the
entities its plans serve on the recipe's deflator role (`read_plan.served_role_entities`),
read from the plan and the coverage index in memory. That holds for an open role too, such
as the donors of "humanitarian aid from donors to Kenya". A donor whose deflator starts
late loses its earlier rows and shortens no other donor's run; the answer names it
(`convertibility.deflator_omissions`).

The offer lists each currency the recipe's exchange-rate table holds with the years each
price basis converts: the longest run of consecutive years, inside the years the answer
serves (`download.window`), whose rows convert (`longest_run`). A unit request whose tables
cover only part of the answer's years shows that run (`conversion_period`), as the offer
states it. The offer also states the base years the tables accept for every served entity
and declare (`RecipeTables.base_year_min` to `base_year_max`), and the national currency of
each role bound to one entity, among the offered currencies.

Every fact comes from the request's `ConversionCatalogue`, loaded once at app startup, and
the answer's read plans, so the offer makes no network read. With no catalogue loaded, an
answer offers nothing, and a currency answer carries `unit_conversion_unavailable`
(`conversion_unavailable_warning`) so that null does not read as "cannot convert".
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final, Literal, Self

from materialise.convert import TermKey, row_terms
from materialise.facets import native_currency_iso
from materialise.recipe_bodies import ConversionRecipe
from materialise.recipes import (
    RecipeNotDeclaredError,
    graph_role_for_spec_role,
    resolve_deflator_role,
)
from search.engine.region.codes import UNIT_CONVERSION_UNAVAILABLE
from search.engine.retrieval.member_facts import MemberFacts
from search.engine.retrieval.read_plan import (
    ReadPlan,
    plan_roles,
    served_facet_keys,
    served_role_entities,
    served_units,
)
from search.engine.unit.conversion_catalogue import (
    ConversionCatalogue,
    RecipeTables,
    conversion_catalogue_unavailable,
    current_conversion_catalogue,
)
from search.models import (
    ConversionCurrency,
    EntityRoleKey,
    IdentityFacetBinding,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    Spec,
    TimeWindow,
    UnitConversion,
    Warning,
)
from unit_words.currency_nouns import currency_noun

type PriceBasis = Literal["current", "constant"]
type YearRun = tuple[int, int]
"""The first and last years of a run of consecutive years."""
type ConversionTable = Literal["exchange_rates", "deflators"]
"""The conversion table that leaves a year out: the exchange rates or the deflators."""

_NOMINAL = "Nominal"


@dataclass(frozen=True)
class ConvertedYears:
    """The years of a set whose rows convert, split by the table that holds their terms:
    `rates`, in which the exchange-rate table holds every rate a row reads, and
    `deflators`, in which the deflator table holds the deflators of at least one entity's
    rows."""

    rates: frozenset[int]
    deflators: frozenset[int]

    @property
    def years(self) -> frozenset[int]:
        """The years whose rows convert: those both tables cover."""
        return self.rates & self.deflators


def _kept_deflator_years(held: frozenset[int], base_year: int | None) -> frozenset[int]:
    """The years whose rows keep their deflator terms, given the years `held` the deflator
    table holds for their key: all of them, or none when they lack `base_year`
    (`materialise.convert` leaves out every row of such an entity). None leaves the
    base-year term unread. The offers (`ConversionAvailability.entity_years`) and the counts
    (`CountedConversion.kept_years`) both read this rule."""
    return held if base_year is None or base_year in held else frozenset()


_NOTHING = ConvertedYears(rates=frozenset(), deflators=frozenset())


@dataclass(frozen=True)
class ConversionAvailability:
    """Which years convert for rows in `natives` deflated through `deflator_entities` under
    one recipe's tables, read only from the tables in memory.

    `native_years` holds the years the exchange-rate table holds a rate for every native
    currency, None when it names no currency, which then bounds no year. `deflator_years`
    holds, for each deflator entity, the years the deflator table holds a value for its key
    (`RecipeTables.deflator_years_of`). A missing rate refuses every row of its year, so the
    rates bound the years for all rows. A missing deflator leaves out only its entity's rows
    (`materialise.convert`), so a year converts while any entity's deflator holds it. Build
    one with `of`.
    """

    tables: RecipeTables
    native_years: frozenset[int] | None
    deflator_years: Mapping[str, frozenset[int]]

    @classmethod
    def of(
        cls,
        tables: RecipeTables,
        recipe: ConversionRecipe,
        *,
        natives: Collection[str],
        deflator_entities: Collection[str],
    ) -> Self:
        """The availability of rows whose native currency is one of `natives` and whose
        entity on the recipe's deflator role is one of `deflator_entities`. A collection
        left empty bounds nothing: no native currency is known, or no deflator entity."""
        return cls(
            tables=tables,
            native_years=_every(tables.rate_years.get(code, frozenset()) for code in natives),
            deflator_years={
                entity: tables.deflator_years_of(entity, fallback=recipe.fallback_entity)
                for entity in deflator_entities
            },
        )

    def entity_years(self, entity: str, *, base_year: int | None) -> frozenset[int]:
        """The years whose rows of `entity`, on the deflator role, keep their deflator terms:
        those the deflator table holds for its key, and none when it lacks `base_year`. None
        leaves the base-year term unread."""
        return _kept_deflator_years(self.deflator_years.get(entity, frozenset()), base_year)

    def converted_years(
        self,
        *,
        years: frozenset[int],
        targets: Collection[str],
        prices: PriceBasis,
        base_year: int | None,
    ) -> ConvertedYears:
        """The years of `years` whose rows convert into `targets` at `prices`: each year in
        which the tables hold every rate term of `materialise.convert.row_terms(prices)` for
        every native currency and target, and at constant prices the deflator terms of at
        least one deflator entity (`entity_years`). A target's rate the tables miss in
        `base_year` leaves no year; None leaves the base-year terms unread, as an offer
        states its years before a base year is picked. Empty `targets` bound nothing, as for
        a role-shaped target with no mapped entity."""
        held = self._rates_held(targets)
        rates = years
        deflators = years
        for term in row_terms(prices):
            if term.table == "deflator":
                continue
            term_years = held[term.key]
            if term_years is None:
                continue
            if term.at_base_year:
                if base_year is not None and base_year not in term_years:
                    return _NOTHING
            else:
                rates = rates & term_years
        if prices == "constant" and self.deflator_years:
            deflators = years & frozenset[int]().union(
                *(self.entity_years(entity, base_year=base_year) for entity in self.deflator_years)
            )
        return ConvertedYears(rates=rates, deflators=deflators)

    def base_years(self, *, targets: Collection[str]) -> frozenset[int]:
        """The base years a constant-price conversion into `targets` accepts: those in which
        the tables hold the target's rate (the rate base-year term of
        `materialise.convert.row_terms("constant")`) for every target and the deflator of at
        least one deflator entity. An entity whose deflator lacks the base year loses its
        rows, so it bounds no base year."""
        held = self._rates_held(targets)
        accepted = self.tables.years
        for term in row_terms("constant"):
            term_years = held.get(term.key)
            if term.at_base_year and term.table == "exchange_rate" and term_years is not None:
                accepted = accepted & term_years
        if self.deflator_years:
            accepted = accepted & frozenset[int]().union(*self.deflator_years.values())
        return accepted

    def _rates_held(self, targets: Collection[str]) -> Mapping[TermKey, frozenset[int] | None]:
        """The years the exchange-rate table holds every key of each rate term key, None for
        no key."""
        return {
            "native": self.native_years,
            "target": _every(self.tables.rate_years.get(code, frozenset()) for code in targets),
        }


def _every(year_sets: Iterable[frozenset[int]]) -> frozenset[int] | None:
    """The years every set holds, None for no set."""
    sets = list(year_sets)
    return frozenset.intersection(*sets) if sets else None


UNDECIDED: Final = "undecided"
"""A unit request whose effect on a count cannot be decided in memory (`counted_conversion`)."""


@dataclass(frozen=True)
class CountedConversion:
    """The rows a unit request's conversion keeps of each series an answer counts.

    A rate the tables lack refuses the conversion, and the period a unit request shows
    (`conversion_period`) already leaves out the years without one, so rates leave no row
    out of a count. A deflator the tables lack leaves out its entity's rows
    (`materialise.convert`): at constant prices a series keeps the years its entity on
    `deflator_role` holds a deflator in, and none when that entity's deflator lacks
    `base_year` (`ConversionAvailability.entity_years`). A conversion into each entity's
    own currency (`target_role`) leaves out every row of an entity no single currency
    names. `natives` are the ISO codes of the currencies the counted rows convert from; a
    series in a facet of any other unit converts under rules this count did not read."""

    tables: RecipeTables
    recipe: ConversionRecipe
    natives: frozenset[str]
    base_year: int | None
    deflator_role: str | None
    target_role: str | None
    named: Collection[str]

    def kept_years(self, entities: Mapping[str, str]) -> frozenset[int] | None:
        """The years a series binding `entities`, by graph role, keeps; None for every year
        it serves. A role `entities` leaves out keeps every year."""
        target = entities.get(self.target_role) if self.target_role is not None else None
        if target is not None and target not in self.named:
            return frozenset()
        entity = entities.get(self.deflator_role) if self.deflator_role is not None else None
        if entity is None:
            return None
        return _kept_deflator_years(
            self.tables.deflator_years_of(entity, fallback=self.recipe.fallback_entity),
            self.base_year,
        )

    def converts(self, unit: str | None) -> bool:
        """Whether rows in a facet of `unit` convert under the rules this count read."""
        return native_currency_iso(unit) in self.natives


type CountConversion = CountedConversion | Literal["undecided"] | None
"""How a unit request changes an answer's counts: the rows its conversion keeps, `UNDECIDED`
when that cannot be read in memory, or None for an answer no conversion changes."""


def pins_facets(facets: Mapping[str, IdentityFacetBinding] | None) -> bool:
    """Whether the identity's `facets` bind a facet value a count does not apply: any
    binding but a unit request, whose effect a count reads (`counted_conversion`)."""
    return any(
        not isinstance(binding, IdentityFacetValueRequest) for binding in (facets or {}).values()
    )


def counted_conversion(
    plan: ReadPlan | None,
    facets: Mapping[str, IdentityFacetBinding] | None,
    *,
    catalogue: ConversionCatalogue | None,
) -> CountConversion:
    """How the unit request among the identity's `facets` changes the counts of the answer
    `plan` serves, read from `catalogue` and the plan in memory.

    None without a unit request, or for one every served facet satisfies natively (a
    current-price request in their own currency). `UNDECIDED` when no catalogue is loaded,
    the served facets name a unit that is no currency or more or fewer than one recipe, or
    the plan's roles place no single deflator role or no role the target names; each of
    those conversions refuses or reads tables per member, so a count cannot follow it."""
    binding = (facets or {}).get("unit")
    if plan is None or not isinstance(binding, IdentityFacetValueRequest):
        return None
    request: IdentityUnitRequest = binding.request
    keys = served_facet_keys(plan)
    natives = {native_currency_iso(unit) for unit, _provenance in keys}
    natively = isinstance(request.currency, str) and natives == {request.currency}
    if not keys or (request.prices == "current" and natively):
        return None
    recipes = {catalogue.recipe(p) if p else None for _u, p in keys} if catalogue else {None}
    recipe = next(iter(recipes)) if len(recipes) == 1 else None
    if catalogue is None or recipe is None or None in natives:
        return UNDECIDED
    roles = plan_roles(plan)
    try:
        deflator_role = (
            resolve_deflator_role(recipe, declared_roles=roles)
            if request.prices == "constant"
            else None
        )
        target_role = (
            None
            if isinstance(request.currency, str)
            else graph_role_for_spec_role(spec_role=request.currency.role, declared_roles=roles)
        )
    except RecipeNotDeclaredError:
        return UNDECIDED
    return CountedConversion(
        tables=catalogue.tables[recipe.recipe_id],
        recipe=recipe,
        natives=frozenset(code for code in natives if code is not None),
        base_year=request.base_year,
        deflator_role=deflator_role,
        target_role=target_role,
        named=catalogue.national_currencies.keys(),
    )


def merged_conversion(conversions: Sequence[CountConversion]) -> CountConversion:
    """The conversion of answers merged into one, whose counts read each answer's: the one
    they share, that one reading every answer's native currencies when only those differ,
    or `UNDECIDED` when they differ otherwise."""
    first = conversions[0] if conversions else None
    if all(conversion == first for conversion in conversions):
        return first
    if not isinstance(first, CountedConversion) or not all(
        isinstance(c, CountedConversion) and replace(c, natives=first.natives) == first
        for c in conversions
    ):
        return UNDECIDED
    natives = frozenset[str]().union(
        *(c.natives for c in conversions if isinstance(c, CountedConversion))
    )
    return replace(first, natives=natives)


def counted_exact(
    exact: bool, conversion: CountConversion, *, units: Iterable[str | None] = ()
) -> bool:
    """Whether a count is exact under `conversion`: `exact`, the count's own exactness over
    the rows the conversion keeps, unless the conversion is `UNDECIDED` or converts rows in
    a facet of one of `units` under rules it did not read (`CountedConversion.converts`).
    `download.rows`, a slot option's and a measure alternative's `served_rows` all state
    their exactness here."""
    if conversion == UNDECIDED:
        return False
    if isinstance(conversion, CountedConversion):
        return exact and all(conversion.converts(unit) for unit in units)
    return exact


def with_unit_conversion(
    spec: Spec, *, members: Sequence[MemberFacts], plans: Sequence[ReadPlan]
) -> Spec:
    """A copy of `spec` stating the conversions it offers under the request's catalogue
    (`current_conversion_catalogue`), or `spec` itself when it offers none."""
    offer = unit_conversion(
        spec, members=members, plans=plans, catalogue=current_conversion_catalogue()
    )
    if offer is None:
        return spec
    resolved = spec.resolved.model_copy(update={"unit_conversion": offer})
    return spec.model_copy(update={"resolved": resolved})


def conversion_unavailable_warning(spec: Spec) -> Warning | None:
    """`unit_conversion_unavailable` for an answer in a currency, or converted by a unit
    request, served while the request's scope binds no loaded catalogue
    (`conversion_catalogue_unavailable`); None otherwise."""
    if not conversion_catalogue_unavailable() or not _states_currency(spec):
        return None
    return Warning(
        code=UNIT_CONVERSION_UNAVAILABLE,
        severity="degraded",
        message=(
            "currency conversion offers are unavailable because the conversion tables have "
            "not loaded; this answer may still convert, so try again shortly."
        ),
    )


def unit_conversion(
    spec: Spec,
    *,
    members: Sequence[MemberFacts],
    plans: Sequence[ReadPlan],
    catalogue: ConversionCatalogue | None,
) -> UnitConversion | None:
    """The conversions `spec` offers, or None when nothing converts its served members.

    Args:
        spec: the built answer; its unit facet, unit request, provenances, entities, and
            download window are read.
        members: the served members' facts, whose five-tuples hold their population type
            and `measurementQualifier`, and whose declared roles place the recipe's
            deflator role (`resolve_deflator_role`).
        plans: the read plans whose rows the answer states, one per answer a same-shape
            merge joins; their served facets give the native currencies
            (`read_plan.served_units`) and their served entities the deflator entities
            (`read_plan.served_role_entities`).
        catalogue: the loaded conversion catalogue; None offers nothing.
    """
    if catalogue is None or not members or not _states_currency(spec):
        return None
    five_tuples = [member.read_five_tuple() for member in members]
    if any(five.meas_qual_dcid != _NOMINAL for five in five_tuples):
        return None
    recipes = {catalogue.recipe(provenance) for provenance in _provenances(spec)}
    recipe = next(iter(recipes)) if len(recipes) == 1 else None
    natives = _natives(spec, plans)
    if (
        recipe is None
        or natives is None
        or any(five.pop_type_dcid not in recipe.flow_population_types for five in five_tuples)
    ):
        return None
    tables = catalogue.tables[recipe.recipe_id]
    served = _served_years(spec, tables)
    deflator_entities = _deflator_entities(recipe, members, plans)
    availability = ConversionAvailability.of(
        tables, recipe, natives=natives, deflator_entities=deflator_entities or ()
    )
    constant = (
        longest_run(
            availability.converted_years(
                years=served, targets=(), prices="constant", base_year=None
            ).years
        )
        if deflator_entities is not None
        else None
    )
    currencies = [
        currency
        for code in tables.currencies
        if (currency := _currency(code, availability, served=served, constant=constant)) is not None
    ]
    if not currencies:
        return None
    offered = {currency.code for currency in currencies}
    return UnitConversion(
        currencies=currencies,
        default_base_year=catalogue.default_base_year,
        role_currencies=_role_currencies(spec, catalogue, currencies=offered),
    )


def longest_run(years: Iterable[int]) -> YearRun | None:
    """The longest run of consecutive years in `years`, the latest of equally long runs, or
    None when `years` is empty."""
    runs: list[YearRun] = []
    for year in sorted(set(years)):
        if runs and year == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], year)
        else:
            runs.append((year, year))
    if not runs:
        return None
    return max(reversed(runs), key=lambda run: run[1] - run[0])


@dataclass(frozen=True)
class ConversionPeriod:
    """The years a unit request shows of an answer whose conversion tables leave some of
    its served years out: `years`, the longest run of years inside `served` whose rows
    convert, and `limited_by`, the table that leaves the other years out (the exchange
    rates whenever they do)."""

    served: YearRun
    years: YearRun
    limited_by: ConversionTable

    @property
    def window(self) -> TimeWindow:
        """`years` as the window the answer reads."""
        return TimeWindow(start_year=self.years[0], end_year=self.years[1])


def conversion_period(*, served: YearRun, converted: ConvertedYears) -> ConversionPeriod | None:
    """The period a unit request shows of an answer serving `served`, given the years whose
    rows convert (`ConversionAvailability.converted_years`). None when they are every
    served year, or none, which the conversion then refuses."""
    needed = frozenset(range(served[0], served[1] + 1))
    run = longest_run(needed & converted.years)
    if run is None or run == served:
        return None
    limited_by: ConversionTable = "deflators" if needed <= converted.rates else "exchange_rates"
    return ConversionPeriod(served=served, years=run, limited_by=limited_by)


def served_prices(spec: Spec) -> PriceBasis | None:
    """The price basis of the unit request the answer's identity binds its unit through,
    served natively or converted. None when no unit request binds it."""
    unit = (spec.definition.identity.facets or {}).get("unit")
    return unit.request.prices if isinstance(unit, IdentityFacetValueRequest) else None


def _unit_requested(spec: Spec) -> bool:
    """Whether the answer's identity binds its unit through a unit request, served natively
    or converted."""
    return served_prices(spec) is not None


def _states_currency(spec: Spec) -> bool:
    """Whether the answer's unit is a currency or a unit request bound it."""
    unit = spec.resolved.facets.get("unit")
    return _unit_requested(spec) or bool(unit and unit.currency)


def _natives(spec: Spec, plans: Sequence[ReadPlan]) -> frozenset[str] | None:
    """The ISO codes of the native currencies of the rows the answer serves: those of the
    facets its plans serve (`served_units`), whether or not a unit request converted them.
    With no served facet, the answer's own unit currency, unless a unit request bound it,
    which leaves the native currency unknown. None when a served unit or the answer's unit
    is no currency, or the native currency is unknown; such an answer offers nothing."""
    units = frozenset[str | None]().union(*(served_units(plan) for plan in plans))
    if units:
        codes = {native_currency_iso(unit) for unit in units}
        return None if None in codes else frozenset(code for code in codes if code is not None)
    unit = spec.resolved.facets.get("unit")
    if _unit_requested(spec) or unit is None or unit.currency is None:
        return None
    return frozenset({unit.currency})


def _served_years(spec: Spec, tables: RecipeTables) -> frozenset[int]:
    """Every year from the first to the last the answer's download serves. Every year the
    tables hold when the answer states no window (it has no read plan, or its rows are
    unknown), which then bounds no run."""
    download = spec.resolved.download
    window = download.window if download is not None else None
    if window is None or window.start_year is None or window.end_year is None:
        return tables.years
    return frozenset(range(window.start_year, window.end_year + 1))


def _deflator_entities(
    recipe: ConversionRecipe, members: Sequence[MemberFacts], plans: Sequence[ReadPlan]
) -> frozenset[str] | None:
    """The entities the answer's plans serve on the recipe's deflator role, placed through
    each member's declared roles (`resolve_deflator_role`), bound or open alike. None when
    a member declares no single deflator role, so a constant-price conversion refuses it
    (`role_undeclared`)."""
    try:
        roles = {
            resolve_deflator_role(recipe, declared_roles=member.read_declared_roles())
            for member in members
        }
    except RecipeNotDeclaredError:
        return None
    return frozenset[str]().union(
        *(served_role_entities(plan, role) for plan in plans for role in roles)
    )


def _currency(
    code: str,
    availability: ConversionAvailability,
    *,
    served: frozenset[int],
    constant: YearRun | None,
) -> ConversionCurrency | None:
    """`code`'s entry: the years a current-price request converts, and its base years, the
    run of consecutive years ending at the latest one the tables accept for `code` and every
    deflator entity (`ConversionAvailability.base_years`) within the years the deflator
    table holds actual values for, with `constant` as its constant-price years when it has
    any. None when neither price basis converts a year."""
    tables = availability.tables
    base_years = sorted(
        year
        for year in availability.base_years(targets=(code,))
        if tables.base_year_min <= year <= tables.base_year_max
    )
    run: list[int] = []
    for year in reversed(base_years):
        if run and year != run[-1] - 1:
            break
        run.append(year)
    current = availability.converted_years(
        years=served, targets=(code,), prices="current", base_year=None
    )
    current_run = longest_run(current.years)
    constant = constant if run else None
    if current_run is None and constant is None:
        return None
    return ConversionCurrency(
        code=code,
        noun=currency_noun(code, label=None),
        current_years=_window(current_run),
        constant_years=_window(constant),
        base_year_min=run[-1] if run else None,
        base_year_max=run[0] if run else None,
    )


def _window(run: YearRun | None) -> TimeWindow | None:
    return TimeWindow(start_year=run[0], end_year=run[1]) if run is not None else None


def _provenances(spec: Spec) -> list[str]:
    """The served provenances: the provenance facet's one ref, else every resolved
    source."""
    provenance = spec.resolved.facets.get("provenance")
    if provenance is not None and provenance.ref is not None:
        return [provenance.ref]
    return [source.ref for source in spec.resolved.trace.resolved_sources]


def _role_currencies(
    spec: Spec, catalogue: ConversionCatalogue, *, currencies: set[str]
) -> dict[EntityRoleKey, str]:
    """The ISO code of each role's one bound entity's national currency, for a role bound
    to exactly one entity whose currency is among ``currencies``."""
    by_role: dict[EntityRoleKey, set[str]] = {}
    for entity in spec.resolved.entities:
        by_role.setdefault(entity.role, set()).add(entity.ref)
    offered: dict[EntityRoleKey, str] = {}
    for role, dcids in by_role.items():
        code = catalogue.national_currencies.get(next(iter(dcids))) if len(dcids) == 1 else None
        if code is not None and code in currencies:
            offered[role] = code
    return offered
