"""The conversions an answer offers (`ResolvedSpec.unit_conversion`) and the catalogue they
come from. The catalogue loads once from the graph and the conversion tables, then reads from memory."""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from types import MappingProxyType, SimpleNamespace
from typing import ClassVar

import pyarrow as pa
import pytest

from code_index.composites import DECLARED_COMPOSITES
from code_index.declarations import CRS_INCLUSIVE
from dc_graph.client import SubjectPage
from materialise.convert import require_conversion_coverage
from materialise.errors import UpstreamResponseError
from materialise.rate_tables import RATE_TABLE_SCHEMA, RateTable, RateTableCache
from materialise.recipe_bodies import RECIPE_BODIES
from materialise.recipes import CONVERSION_NODE_PROPERTIES, resolve_recipe
from search.engine.clients.graph import Facet
from search.engine.conjoin import assemble_region
from search.engine.envelope import now_ms
from search.engine.retrieval.facts import FiveTuple
from search.engine.retrieval.read_plan import PairPlan, PlanSeries, ReadPlan
from search.engine.unit.base_year import TABLE_PROVENANCES
from search.engine.unit.conversion import (
    conversion_unavailable_warning,
    unit_conversion,
    with_unit_conversion,
)
from search.engine.unit.conversion_catalogue import (
    ConversionCatalogue,
    ConversionCatalogueService,
    RecipeTables,
    bound_conversion_catalogue,
    load_conversion_catalogue,
)
from search.models import (
    ConversionCurrency,
    DownloadFacts,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    ResolvedFacet,
    Spec,
    TimeWindow,
)
from tests.engine._builders import conjoin_spec
from tests.engine._conjoin_fixtures import _PERSON_SHAPE
from tests.engine._conjoin_fixtures import _region as conjoin_region
from tests.render._helpers import _entity_from, _entity_subject, _entity_to, _ref, _spec

_REFRESH = "2026-09-01"
_CRS = "provenance/OECD_CRS"
_IDS = "provenance/WB_IDS"
_WEO = "provenance/IMF_WEO"
_NO_RECIPE = "provenance/Undeclared"
_DATASET = "dc/d/ImputedSectors"
_IN_DATASET = "provenance/ImputedSectors"


def _years(first: int, last: int) -> frozenset[int]:
    return frozenset(range(first, last + 1))


def _tables(first: int) -> RecipeTables:
    """Rates for euros, quetzales and US dollars over 1960 to 2031, but quetzales only
    from 1980 and euros missing 2022, under a deflator with actual values from `first` to
    2025, Germany's from 1970 and the fallback's from `first`."""
    return RecipeTables(
        rate_years={
            "EUR": _years(1960, 2031) - {2022},
            "GTQ": _years(1980, 2031),
            "USD": _years(1960, 2031),
        },
        deflator_years={
            "country/DEU": _years(1970, 2026),
            "DACCountriesTotal": _years(first, 2026),
        },
        base_year_min=first,
        base_year_max=2025,
    )


def _run(first: int, last: int) -> TimeWindow:
    return TimeWindow(start_year=first, end_year=last)


_CATALOGUE = ConversionCatalogue(
    recipes=MappingProxyType(
        {_CRS: "ConversionRecipeDAC", _IDS: "ConversionRecipeIDS", _WEO: "ConversionRecipeIMF"}
    ),
    tables=MappingProxyType(
        {
            "ConversionRecipeDAC": _tables(1960),
            "ConversionRecipeIDS": _tables(1980),
            "ConversionRecipeIMF": _tables(1980),
        }
    ),
    national_currencies=MappingProxyType({"country/DEU": "EUR", "country/GTM": "GTQ"}),
    default_base_year=2024,
)


def _members(
    population_type: str = "FinancialFlow",
    qualifier: str | None = "Nominal",
    roles: tuple[str, ...] = ("observationAbout",),
):
    five = FiveTuple(population_type, "amount", "measuredValue", qualifier, None)
    return [SimpleNamespace(read_five_tuple=lambda: five, read_declared_roles=lambda: roles)]


def _usd(provenance: str = _CRS, *, entities=None, served: tuple[int, int] | None = None) -> Spec:
    spec = _spec(
        entities=entities,
        facets={
            "unit": ResolvedFacet(basis="fallback", ref="CurrencyUSD", currency="USD"),
            "provenance": ResolvedFacet(basis="fallback", ref=provenance),
        },
    )
    if served is None:
        return spec
    download = DownloadFacts(
        rows=1,
        rows_exact=True,
        series=1,
        window=TimeWindow(start_year=served[0], end_year=served[1]),
    )
    return spec.model_copy(
        update={"resolved": spec.resolved.model_copy(update={"download": download})}
    )


type _RoleTuple = tuple[tuple[str, str], ...]

_DONORS = ("financingProvider", "financingRecipient")


def _plan(
    served: tuple[int, int],
    *tuples: _RoleTuple,
    provenance: str = _CRS,
    unit: str = "CurrencyUSD",
) -> PairPlan:
    """A plan serving one series per role tuple, each with one point a year over `served`
    in `unit`, the facet's native unit."""
    years = tuple(str(year) for year in range(served[0], served[1] + 1))
    facet = Facet(
        obs_count=len(years),
        earliest_date=years[0],
        latest_date=years[-1],
        facet_id="f1",
        provenance_id=provenance,
        unit=unit,
        dates=years,
    )
    series = tuple(
        PlanSeries(stat_var="TEST/Flow", role_tuple=roles, candidates=(facet,), group=(facet,))
        for roles in tuples
    )
    return PairPlan(members=("TEST/Flow",), years=None, series=series)


def _offer(spec: Spec, members=None, plans: tuple[ReadPlan, ...] = ()):
    members = members if members is not None else _members()
    return unit_conversion(spec, members=members, plans=plans, catalogue=_CATALOGUE)


@pytest.mark.parametrize(
    ("provenance", "first_year", "roles", "entity"),
    [
        # Guatemala has no row in the deflator table, so it reads the DAC fallback's.
        (_CRS, 1960, ("observationAbout",), "country/GTM"),
        # Debt deflates by the debtor's prices.
        (_IDS, 1980, ("financingRecipient",), "country/DEU"),
    ],
    ids=["crs", "ids"],
)
def test_a_flow_under_a_recipe_offers_its_tables_currencies_and_years(
    provenance, first_year, roles, entity
):
    plan = _plan((1990, 2024), ((roles[0], entity),), provenance=provenance)
    offer = _offer(_usd(provenance, served=(1990, 2024)), _members(roles=roles), (plan,))
    assert offer is not None
    assert offer.default_base_year == 2024
    served = _run(1990, 2024)
    assert offer.currencies == [
        ConversionCurrency(
            code="EUR",
            noun="euros",
            # Euro rates miss 2022, so the longest run ends in 2021.
            current_years=_run(1990, 2021),
            constant_years=served,
            base_year_min=2023,
            base_year_max=2025,
        ),
        ConversionCurrency(
            code="GTQ",
            noun="Guatemalan quetzales",
            current_years=served,
            constant_years=served,
            base_year_min=max(first_year, 1980),
            base_year_max=2025,
        ),
        ConversionCurrency(
            code="USD",
            noun="US dollars",
            current_years=served,
            constant_years=served,
            base_year_min=first_year,
            base_year_max=2025,
        ),
    ]


def test_a_currency_whose_rates_start_after_the_served_years_offers_the_years_they_cover():
    """CRS aid to Guatemala runs from 1976 and the quetzal's rates from 1980, so a
    current-price request in quetzales shows 1980 to 2024."""
    plan = _plan((1976, 2024), (("observationAbout", "country/GTM"),))
    offer = _offer(_usd(served=(1976, 2024)), plans=(plan,))
    assert offer is not None
    gtq = next(currency for currency in offer.currencies if currency.code == "GTQ")
    assert (gtq.current_years, gtq.constant_years, gtq.base_year_min) == (
        _run(1980, 2024),
        _run(1976, 2024),
        1980,
    )


_GERMAN_AID = _plan((1965, 2024), (("financingProvider", "country/DEU"),))
_DONORS_AID = _plan(
    (1965, 2024),
    (("financingProvider", "country/DEU"),),
    # France has no row in the deflator table, so it reads the DAC fallback's, from 1960.
    (("financingProvider", "country/FRA"),),
)


@pytest.mark.parametrize(
    ("entities", "plan", "first"),
    [
        pytest.param(
            [_entity_from("country/DEU", "Germany")], _GERMAN_AID, (1970, 1970), id="bound-donor"
        ),
        pytest.param(None, _DONORS_AID, (1965, 1960), id="open-donors"),
    ],
)
def test_constant_years_and_base_years_follow_the_donor_whose_deflator_holds_them(
    entities, plan, first
):
    """Germany's deflator starts in 1970, so German aid from 1965 converts at constant
    prices from 1970, in a base year from 1970. With France among the donors, whose
    fallback deflator runs from 1960, Germany's earlier rows are left out and France's
    convert: the run starts in 1965, the first served year, and the base years in 1960."""
    first_year, first_base_year = first
    offer = _offer(_usd(entities=entities, served=(1965, 2024)), _members(roles=_DONORS), (plan,))
    assert offer is not None
    assert [(c.code, c.constant_years, c.base_year_min) for c in offer.currencies] == [
        ("EUR", _run(first_year, 2024), 2023),
        ("GTQ", _run(first_year, 2024), max(first_base_year, 1980)),
        ("USD", _run(first_year, 2024), first_base_year),
    ]


def test_a_native_currency_without_rates_in_a_served_year_offers_the_years_it_has():
    """IDS debt service runs from 1970 and its tables' dollar rates from 1980. Every
    conversion divides by the dollar's rate in each year, so every price basis shows 1980
    to 2024. A unit request the dollar series serve natively leaves their rows in dollars,
    so the offer it then states still divides by the dollar's rates."""
    offers = [
        unit_conversion(spec, members=_IDS_MEMBERS, plans=(_IDS_PLAN,), catalogue=_IDS_CATALOGUE)
        for spec in (_usd(_IDS, served=(1970, 2024)), _requested(_usd(_IDS, served=(1970, 2024))))
    ]
    assert [
        [(c.code, c.current_years, c.constant_years) for c in o.currencies] for o in offers
    ] == [
        [("EUR", _run(1980, 2024), _run(1980, 2024)), ("USD", _run(1980, 2024), _run(1980, 2024))]
    ] * 2


_IDS_CATALOGUE = ConversionCatalogue(
    recipes=_CATALOGUE.recipes,
    tables={
        **_CATALOGUE.tables,
        "ConversionRecipeIDS": RecipeTables(
            rate_years={"EUR": _years(1960, 2031), "USD": _years(1980, 2031)},
            deflator_years={"country/GTM": _years(1960, 2026)},
            base_year_min=1980,
            base_year_max=2025,
        ),
    },
    national_currencies=_CATALOGUE.national_currencies,
    default_base_year=2024,
)
"""IDS tables whose euro rates run from 1960 and dollar rates from 1980."""
_IDS_PLAN = _plan((1970, 2024), (("financingRecipient", "country/GTM"),), provenance=_IDS)
_IDS_MEMBERS = _members(roles=("financingRecipient",))


def _requested(spec: Spec, request: IdentityUnitRequest | None = None) -> Spec:
    """`spec` with its identity binding its unit through `request`, current US dollars by
    default."""
    request = request or IdentityUnitRequest(currency="USD", prices="current")
    identity = spec.definition.identity.model_copy(
        update={"facets": {"unit": IdentityFacetValueRequest(request=request)}}
    )
    definition = spec.definition.model_copy(update={"identity": identity})
    return spec.model_copy(update={"definition": definition})


def test_a_converted_answer_still_offers_conversion():
    """An answer already in constant 2024 euros offers the same conversions, from the
    dollars its series are served in."""
    request = IdentityUnitRequest(currency="EUR", prices="constant", base_year=2024)
    spec = _spec(
        facets={"unit": ResolvedFacet(basis="bound", ref="CurrencyEUR", currency="EUR")},
        resolved_sources=[_ref(_CRS, "CRS")],
        identity_facets={"unit": IdentityFacetValueRequest(request=request)},
    )
    plan = _plan((1990, 2024), (("observationAbout", "country/GTM"),))
    assert _offer(spec, plans=(plan,)) is not None
    assert _offer(spec) is None, "a unit request leaves the native currency to the plan"


# --- The conversion accepts every offer -------------------------------------------------


def _rate_tables(catalogue: ConversionCatalogue, recipe_id: str) -> tuple[RateTable, RateTable]:
    """The exchange-rate and deflator tables behind `catalogue`'s `recipe_id` tables, with
    an actual value in every year they hold."""
    recipe = RECIPE_BODIES[recipe_id]
    tables = catalogue.tables[recipe_id]
    rates = [
        (f"Currency{code}", year, "actual")
        for code, years in tables.rate_years.items()
        for year in years
    ]
    deflators = [
        (entity, year, "actual")
        for entity, years in tables.deflator_years.items()
        for year in years
    ]
    provenance = recipe.table_provenance
    return (
        _table("exchange_rate", recipe.exchange_rate_variable, provenance, rates),
        _table("deflator", recipe.deflator_variable, provenance, deflators),
    )


@pytest.mark.parametrize(
    ("spec", "members", "plan", "catalogue", "recipe_id", "donors"),
    [
        pytest.param(
            _usd(entities=[_entity_from("country/DEU", "Germany")], served=(1965, 2024)),
            _members(roles=_DONORS),
            _GERMAN_AID,
            _CATALOGUE,
            "ConversionRecipeDAC",
            ("country/DEU",),
            id="german-aid",
        ),
        pytest.param(
            _usd(served=(1965, 2024)),
            _members(roles=_DONORS),
            _DONORS_AID,
            _CATALOGUE,
            "ConversionRecipeDAC",
            ("country/DEU", "country/FRA"),
            id="open-donors",
        ),
        pytest.param(
            _requested(_usd(_IDS, served=(1970, 2024))),
            _IDS_MEMBERS,
            _IDS_PLAN,
            _IDS_CATALOGUE,
            "ConversionRecipeIDS",
            ("country/GTM",),
            id="ids-native-dollar-request",
        ),
    ],
)
def test_the_conversion_accepts_every_year_and_base_year_the_offer_states(
    spec, members, plan, catalogue, recipe_id, donors
):
    """Each offered currency, price basis and base year converts every row the offer's
    years hold, at every entity the plan serves on the deflator role, under the check a
    conversion runs (`require_conversion_coverage`)."""
    offer = unit_conversion(spec, members=members, plans=(plan,), catalogue=catalogue)
    assert offer is not None
    recipe = RECIPE_BODIES[recipe_id]
    exchange_rates, deflators = _rate_tables(catalogue, recipe_id)
    checked = 0
    for currency in offer.currencies:
        target = f"Currency{currency.code}"
        if (window := currency.current_years) is not None:
            years = list(range(window.start_year, window.end_year + 1))
            require_conversion_coverage(
                prices="current",
                base_year=None,
                recipe=recipe,
                native_currency="CurrencyUSD",
                target_currencies=[target] * len(years),
                deflator_entities=None,
                years=years,
                exchange_rate_table=exchange_rates,
            )
            checked += 1
        if (window := currency.constant_years) is None:
            continue
        assert currency.base_year_min is not None and currency.base_year_max is not None
        rows = [
            (year, donor)
            for year in range(window.start_year, window.end_year + 1)
            for donor in donors
        ]
        for base_year in range(currency.base_year_min, currency.base_year_max + 1):
            require_conversion_coverage(
                prices="constant",
                base_year=base_year,
                recipe=recipe,
                native_currency="CurrencyUSD",
                target_currencies=[target] * len(rows),
                deflator_entities=[donor for _year, donor in rows],
                years=[year for year, _donor in rows],
                exchange_rate_table=exchange_rates,
                deflator_table=deflators,
            )
            checked += 1
    assert checked > len(offer.currencies)


@pytest.mark.parametrize(
    ("spec", "members"),
    [
        pytest.param(_usd(), _members("FinancialPosition"), id="stock"),
        pytest.param(_usd(), _members(qualifier="RealValue"), id="constant-prices-series"),
        pytest.param(_usd(_NO_RECIPE), _members(), id="provenance-without-recipe"),
        pytest.param(
            _spec(facets={"unit": ResolvedFacet(basis="fallback", ref="Percent")}),
            _members(),
            id="not-a-currency",
        ),
    ],
)
def test_no_recipe_offers_nothing(spec: Spec, members):
    assert _offer(spec, members) is None


def test_a_composite_offers_the_recipe_its_component_tables_share():
    """A composite answer's provenance facet names the composite, which names no recipe; its
    component tables do, and they share one."""
    composite = DECLARED_COMPOSITES[CRS_INCLUSIVE]
    catalogue = ConversionCatalogue(
        recipes=MappingProxyType(
            {recipe.dataset: "ConversionRecipeDAC" for recipe in composite.components}
        ),
        tables=_CATALOGUE.tables,
        national_currencies=_CATALOGUE.national_currencies,
        default_base_year=2024,
    )
    plan = _plan((1990, 2024), (("observationAbout", "country/GTM"),))

    offer = unit_conversion(
        _usd(composite.provenance, served=(1990, 2024)),
        members=_members(),
        plans=(plan,),
        catalogue=catalogue,
    )

    assert offer is not None
    assert [currency.code for currency in offer.currencies] == ["EUR", "GTQ", "USD"]


def test_a_role_bound_to_one_entity_offers_its_national_currency():
    offer = _offer(
        _usd(entities=[_entity_from("country/DEU", "Germany"), _entity_to("country/GTM", "G")])
    )
    assert offer is not None
    assert offer.role_currencies == {"origin": "EUR", "destination": "GTQ"}


def test_a_role_currency_the_offer_leaves_out_is_absent():
    """A currency with no rate in any base year or served year is no offer to name."""
    tables = RecipeTables(
        rate_years={"EUR": _years(1960, 1970), "USD": _years(1960, 2031)},
        deflator_years={},
        base_year_min=2000,
        base_year_max=2025,
    )
    catalogue = ConversionCatalogue(
        recipes=_CATALOGUE.recipes,
        tables={**_CATALOGUE.tables, "ConversionRecipeDAC": tables},
        national_currencies=_CATALOGUE.national_currencies,
        default_base_year=2024,
    )
    spec = _usd(entities=[_entity_from("country/DEU", "Germany")], served=(1990, 2024))
    offer = unit_conversion(spec, members=_members(), plans=(), catalogue=catalogue)
    assert offer is not None
    assert [currency.code for currency in offer.currencies] == ["USD"]
    assert offer.role_currencies == {}


def test_a_role_bound_to_a_group_offers_no_currency():
    """The 27 EU members are no single entity to name a currency for, so origin is absent."""
    members = [_entity_from(f"country/{code}", code) for code in ("DEU", "FRA", "ITA")]
    offer = _offer(_usd(entities=[*members, _entity_to("country/GTM", "Guatemala")]))
    assert offer is not None
    assert offer.role_currencies == {"destination": "GTQ"}


def test_an_entity_with_no_national_currency_is_absent():
    spec = _usd(_WEO, entities=[_entity_subject("africa", "Africa")])
    offer = _offer(spec, _members("EconomicProduction"))
    assert offer is not None
    assert offer.role_currencies == {}


def test_an_answer_offers_nothing_without_a_bound_catalogue():
    spec = _usd()
    assert with_unit_conversion(spec, members=_members(), plans=()).resolved.unit_conversion is None
    with bound_conversion_catalogue(_CATALOGUE):
        assert with_unit_conversion(spec, members=_members(), plans=()).resolved.unit_conversion


@pytest.mark.parametrize(
    ("spec", "bound", "warns"),
    [
        pytest.param(_usd(), None, True, id="currency-no-catalogue-loaded"),
        pytest.param(_usd(), _CATALOGUE, False, id="currency-catalogue-loaded"),
        pytest.param(
            _spec(facets={"unit": ResolvedFacet(basis="fallback", ref="Percent")}),
            None,
            False,
            id="not-a-currency",
        ),
    ],
)
def test_a_currency_answer_warns_while_no_catalogue_is_loaded(spec: Spec, bound, warns):
    """A null offer from an app that has loaded no catalogue reflects a missing load, so the
    part says so."""
    with bound_conversion_catalogue(bound):
        warning = conversion_unavailable_warning(spec)
    assert (warning.code if warning is not None else None) == (
        "unit_conversion_unavailable" if warns else None
    )


def test_no_warning_outside_a_request_scope():
    assert conversion_unavailable_warning(_usd()) is None


@pytest.mark.parametrize(("bound", "warns"), [(None, True), (_CATALOGUE, False)])
def test_a_definite_dollar_part_carries_the_warning_while_no_catalogue_is_loaded(bound, warns):
    """The part assembled for a dollar answer, as the app's request scope binds the service's
    catalogue or, before any load succeeds, none."""
    spec = conjoin_spec(_PERSON_SHAPE, "any")
    unit = ResolvedFacet(basis="fallback", ref="CurrencyUSD", currency="USD")
    facets = {**spec.resolved.facets, "unit": unit}
    spec = spec.model_copy(update={"resolved": spec.resolved.model_copy(update={"facets": facets})})
    region = conjoin_region(spec, "aid")
    with bound_conversion_catalogue(bound):
        response = assemble_region(
            region,
            query="aid",
            extra_warnings=[],
            start_ms=now_ms(),
            engine_build="test",
            max_candidates=10,
        )
    codes = [warning.code for warning in response.parts[0].warnings]
    assert ("unit_conversion_unavailable" in codes) is warns


# --- Loading the catalogue -------------------------------------------------------------


class _Graph:
    """A graph holding the recipe each provenance or Dataset names, the Datasets each
    provenance is part of, the conversion tables' refresh date and base year, and the
    currency units with the places they serve. `extra` adds or replaces nodes."""

    _CURRENCIES: ClassVar[dict[str, tuple[str, ...]]] = {
        "CurrencyEUR": ("country/DEU", "country/FRA", "country/MNE"),
        "CurrencyGTQ": ("country/GTM",),
        # Montenegro is named by two currencies, so it has none.
        "CurrencyXXX": ("country/MNE",),
    }

    def __init__(self, extra: Mapping[str, Mapping[str, tuple[str, ...]]] | None = None) -> None:
        self._nodes: dict[str, Mapping[str, tuple[str, ...]]] = {
            _CRS: {"typeOf": ("Provenance",), "conversionRecipe": ("ConversionRecipeDAC",)},
            _DATASET: {"typeOf": ("Dataset",), "conversionRecipe": ("ConversionRecipeDAC",)},
            _IN_DATASET: {"typeOf": ("Provenance",), "isPartOf": (_DATASET,)},
            _IDS: {"typeOf": ("Provenance",), "conversionRecipe": ("ConversionRecipeIDS",)},
            _WEO: {"typeOf": ("Provenance",), "conversionRecipe": ("ConversionRecipeIMF",)},
            **{
                dcid: {"lastDataRefreshDate": (_REFRESH,), "deflatorBaseYear": ("2024",)}
                for dcid in TABLE_PROVENANCES
            },
            **{
                unit: {"typeOf": ("CurrencyUnit",), "nationalCurrencyOf": places}
                for unit, places in self._CURRENCIES.items()
            },
            "DACCountriesTotal": {"name": ("DAC countries, total",)},
            "country/GTM": {"name": ("Guatemala",)},
            **(extra or {}),
        }

    def node_arcs_many(
        self, dcids: Sequence[str], *, properties: Sequence[str], deadline: float | None = None
    ) -> dict[str, Mapping[str, tuple[str, ...]] | None]:
        arcs: dict[str, Mapping[str, tuple[str, ...]] | None] = {}
        for dcid in dcids:
            node = self._nodes.get(dcid, {"typeOf": ("Provenance",)})
            arcs[dcid] = {prop: node[prop] for prop in properties if prop in node} or None
        return arcs

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> SubjectPage:
        subjects = tuple(
            subject for subject, node in self._nodes.items() if dcid in node.get(property, ())
        )
        return SubjectPage(dcids=subjects, truncated=False)


def _table(kind, stat_var: str, provenance: str, rows: list[tuple[str, int, str]]) -> RateTable:
    table = pa.table(
        {
            "entity": [entity for entity, _, _ in rows],
            "year": [year for _, year, _ in rows],
            "value": [1.0] * len(rows),
            "facet": [facet for _, _, facet in rows],
        },
        schema=RATE_TABLE_SCHEMA,
    )
    return RateTable(kind, stat_var, provenance, f"v-{stat_var}", _REFRESH, table)


def _cache() -> RateTableCache:
    """Every recipe's tables: rates for US dollars, euros, quetzales, and Zimbabwe gold,
    which no unit request accepts; deflators with actual values from 1960 to 2025 and a
    projection for 2026."""
    cache = RateTableCache()
    for recipe in RECIPE_BODIES.values():
        rates = [(f"Currency{code}", 2020, "actual") for code in ("USD", "EUR", "GTQ", "ZWG")]
        deflators = [
            ("DACCountriesTotal", 1960, "actual"),
            ("DACCountriesTotal", 2025, "actual"),
            ("DACCountriesTotal", 2026, "projected"),
        ]
        provenance = recipe.table_provenance
        cache.admit(_table("exchange_rate", recipe.exchange_rate_variable, provenance, rates))
        cache.admit(_table("deflator", recipe.deflator_variable, provenance, deflators))
    return cache


def _load(graph: _Graph | None = None) -> ConversionCatalogue:
    graph = graph if graph is not None else _Graph()
    return load_conversion_catalogue(graph=graph, reader=graph, sdmx=None, cache=_cache())


def test_the_catalogue_reads_each_provenances_recipe_from_it_or_its_dataset():
    assert dict(_load().recipes) == {
        _CRS: "ConversionRecipeDAC",
        _IN_DATASET: "ConversionRecipeDAC",
        _IDS: "ConversionRecipeIDS",
        _WEO: "ConversionRecipeIMF",
    }


_IDS_DATASET = "dc/d/DebtStatistics"
_IN_TWO = "provenance/InTwoDatasets"


@pytest.mark.parametrize(
    ("datasets", "first"),
    [
        ((_DATASET, _IDS_DATASET), "ConversionRecipeDAC"),
        ((_IDS_DATASET, _DATASET), "ConversionRecipeIDS"),
    ],
)
def test_a_provenance_in_two_datasets_takes_the_first_ones_recipe_as_the_materialiser_does(
    datasets: tuple[str, str], first: str
):
    graph = _Graph(
        {
            _IDS_DATASET: {"typeOf": ("Dataset",), "conversionRecipe": ("ConversionRecipeIDS",)},
            _IN_TWO: {"typeOf": ("Provenance",), "isPartOf": datasets},
        }
    )
    arcs = graph.node_arcs_many([_IN_TWO], properties=CONVERSION_NODE_PROPERTIES)[_IN_TWO]
    materialised = resolve_recipe(graph, provenance_id=_IN_TWO, provenance_arcs=arcs)
    assert materialised.recipe_id == first
    assert _load(graph).recipe(_IN_TWO) is materialised


@pytest.mark.parametrize(
    "extra",
    [
        # The provenance names two recipes: the materialiser refuses it, so the catalogue
        # offers none rather than its Dataset's.
        {
            _IN_DATASET: {
                "isPartOf": (_DATASET,),
                "conversionRecipe": ("ConversionRecipeDAC", "ConversionRecipeIMF"),
            }
        },
        # Its first Dataset names a recipe this deployment has no body for.
        {
            "dc/d/Ghost": {"typeOf": ("Dataset",), "conversionRecipe": ("recipe/Ghost",)},
            _IN_DATASET: {"isPartOf": ("dc/d/Ghost", _DATASET)},
        },
    ],
)
def test_the_catalogue_offers_no_recipe_for_a_provenance_the_materialiser_refuses(extra):
    graph = _Graph(extra)
    arcs = graph.node_arcs_many([_IN_DATASET], properties=CONVERSION_NODE_PROPERTIES)[_IN_DATASET]
    with pytest.raises(UpstreamResponseError):
        resolve_recipe(graph, provenance_id=_IN_DATASET, provenance_arcs=arcs)
    assert _IN_DATASET not in _load(graph).recipes


def test_the_catalogue_holds_the_tables_accepted_currencies_and_actual_years():
    catalogue = _load()
    assert catalogue.tables["ConversionRecipeDAC"] == RecipeTables(
        rate_years={code: frozenset({2020}) for code in ("EUR", "GTQ", "USD")},
        deflator_years={"DACCountriesTotal": frozenset({1960, 2025, 2026})},
        base_year_min=1960,
        base_year_max=2025,
    )
    assert catalogue.default_base_year == 2024
    # Guatemala has a national currency and no deflator row: a constant-price answer that
    # leaves out its rows still names it.
    assert dict(catalogue.entity_names) == {
        "DACCountriesTotal": "DAC countries, total",
        "country/GTM": "Guatemala",
    }


def test_a_place_two_currencies_name_has_no_national_currency():
    assert dict(_load().national_currencies) == {
        "country/DEU": "EUR",
        "country/FRA": "EUR",
        "country/GTM": "GTQ",
    }


# --- The loading service ---------------------------------------------------------------


def _service() -> ConversionCatalogueService:
    return ConversionCatalogueService(retry_delays_s=(0.01,))


def _until_loaded(service: ConversionCatalogueService, catalogue: ConversionCatalogue) -> bool:
    """Whether `service` publishes `catalogue` within five seconds."""
    for _ in range(500):
        if service.snapshot() is catalogue:
            return True
        threading.Event().wait(0.01)
    return False


def test_a_failed_load_logs_an_error_and_is_retried_until_one_succeeds(caplog):
    service = _service()
    attempts: list[int] = []

    def flaky() -> ConversionCatalogue:
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("graph unavailable")
        return _CATALOGUE

    # The app's `search` logger may stop propagating, so attach the capture directly.
    logger = logging.getLogger("search.engine.unit.conversion_catalogue")
    logger.addHandler(caplog.handler)
    try:
        service.start(flaky)
        # The first load ends, failed, without waiting out its bound.
        assert not service.wait_first_load(5)
        assert _until_loaded(service, _CATALOGUE)
    finally:
        logger.removeHandler(caplog.handler)
        service.close()
    assert len(attempts) == 3
    assert "conversion catalogue load failed" in caplog.text


def test_a_failed_reload_keeps_the_loaded_catalogue():
    service = _service()
    service.start(lambda: _CATALOGUE)
    assert service.wait_first_load(5)
    failed = threading.Event()

    def fail() -> ConversionCatalogue:
        failed.set()
        raise RuntimeError("graph unavailable")

    service.start(fail)
    assert failed.wait(5)
    service.close()
    assert service.snapshot() is _CATALOGUE


def test_a_reload_requested_during_a_load_runs_after_it_and_the_newest_wins():
    """The first load blocks. Two reloads requested meanwhile run once, after it, so the
    catalogue it publishes is the newest generation's."""
    service = _service()
    release = threading.Event()
    running = threading.Event()
    loads: list[str] = []
    stale = ConversionCatalogue(
        recipes={}, tables={}, national_currencies={}, default_base_year=2023
    )

    def load() -> ConversionCatalogue:
        loads.append("load")
        if len(loads) == 1:
            running.set()
            assert release.wait(5)
            return stale
        return _CATALOGUE

    service.start(load)
    assert running.wait(5)
    service.reload_in_background()
    service.reload_in_background()
    release.set()
    assert _until_loaded(service, _CATALOGUE)
    service.close()
    assert loads == ["load", "load"]


def test_waiting_for_a_load_that_has_not_finished_returns_false_at_its_bound():
    service = _service()
    release = threading.Event()
    service.start(lambda: (release.wait(5), _CATALOGUE)[1])
    assert not service.wait_first_load(0.05)
    release.set()
    assert service.wait_first_load(5)
    service.close()
