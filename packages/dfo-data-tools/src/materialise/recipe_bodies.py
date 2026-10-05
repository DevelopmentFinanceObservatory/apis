"""Deployment configuration for the currency-conversion recipes.

A Provenance names its recipe with one `conversionRecipe` arc whose value is
a member of the `ConversionRecipe` enumeration on the graph. `resolve_recipe`
maps that member to one of these bodies. An unknown member is a deployment
configuration error.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from dfo_spec_kit.identity import canonicalize


@dataclass(frozen=True)
class ConversionRecipe:
    """One currency-conversion recipe.

    `exchange_rate_variable` is a StatVar whose observations are about
    `CurrencyUnit` nodes, in units of the currency per US dollar.
    `deflator_variable` is a US-dollar deflator StatVar whose observations
    are about countries and aggregates, each series rebased to 100 in one
    base year. Both are read under `table_provenance`.

    `deflator_entity_roles` is the unordered set of graph roles that may
    supply a row's deflator entity. `flow_population_types` is the set of
    StatVar population types the recipe converts. Any other population type,
    stocks among them, refuses. `fallback_entity` stands in for a deflator
    entity absent from the whole deflator table.
    """

    recipe_id: str
    exchange_rate_variable: str
    deflator_variable: str
    deflator_entity_roles: frozenset[str]
    flow_population_types: frozenset[str]
    forecast_policy: str
    table_provenance: str
    fallback_entity: str | None = None


def recipe_body_digest(recipe: ConversionRecipe) -> str:
    """SHA-256 hex digest of recipe fields in the same canonical JCS form as
    content-version minting. Sets become sorted arrays and `None` becomes
    JSON null.
    """
    projection: dict[str, object] = {
        "recipe_id": recipe.recipe_id,
        "exchange_rate_variable": recipe.exchange_rate_variable,
        "deflator_variable": recipe.deflator_variable,
        "deflator_entity_roles": sorted(recipe.deflator_entity_roles),
        "flow_population_types": sorted(recipe.flow_population_types),
        "forecast_policy": recipe.forecast_policy,
        "table_provenance": recipe.table_provenance,
        "fallback_entity": recipe.fallback_entity,
    }
    return hashlib.sha256(canonicalize(projection)).hexdigest()


_DAC_TABLES = "provenance/ONEConversionTablesDAC"
_IMF_TABLES = "provenance/ONEConversionTablesIMF"

# The population types of the monetary StatVars CRS, imputed multilateral
# sectors and IDS declare. IDS also types reserves, a stock, and GNP, a flow,
# as `EconomicActivity`, so that type stays out until the pipeline separates
# them.
_FINANCE_FLOW_POPULATION_TYPES = frozenset({"FinancialFlow", "FinancialObligation"})

RECIPE_BODIES: dict[str, ConversionRecipe] = {
    "ConversionRecipeDAC": ConversionRecipe(
        recipe_id="ConversionRecipeDAC",
        exchange_rate_variable="ExchangeRate_CurrencyUnit_ConversionRecipeDAC",
        deflator_variable="Deflator_EconomicProduction_ConversionRecipeDAC",
        deflator_entity_roles=frozenset({"financingProvider", "observationAbout"}),
        flow_population_types=_FINANCE_FLOW_POPULATION_TYPES,
        forecast_policy="FlagProjected",
        table_provenance=_DAC_TABLES,
        fallback_entity="DACCountriesTotal",
    ),
    "ConversionRecipeIMF": ConversionRecipe(
        recipe_id="ConversionRecipeIMF",
        exchange_rate_variable="ExchangeRate_CurrencyUnit_ConversionRecipeIMF",
        deflator_variable="Deflator_EconomicProduction_ConversionRecipeIMF",
        deflator_entity_roles=frozenset({"financingProvider", "observationAbout"}),
        # The population type the WEO import gives its nominal GDP StatVars
        # (`weo_indicator_mapping.csv`).
        flow_population_types=frozenset({"EconomicProduction"}),
        forecast_policy="FlagProjected",
        table_provenance=_IMF_TABLES,
    ),
    "ConversionRecipeIDS": ConversionRecipe(
        recipe_id="ConversionRecipeIDS",
        exchange_rate_variable="ExchangeRate_CurrencyUnit_ConversionRecipeIMF",
        deflator_variable="Deflator_EconomicProduction_ConversionRecipeIMF",
        # Debt deflates by the debtor's prices, and every IDS StatVar
        # declares the debtor role.
        deflator_entity_roles=frozenset({"financingRecipient"}),
        flow_population_types=_FINANCE_FLOW_POPULATION_TYPES,
        forecast_policy="FlagProjected",
        table_provenance=_IMF_TABLES,
    ),
}

RECIPE_CLASSES: tuple[str, ...] = tuple(sorted(RECIPE_BODIES))
"""The recipe members this deployment can apply, published as the
capabilities field `recipe_classes`."""
