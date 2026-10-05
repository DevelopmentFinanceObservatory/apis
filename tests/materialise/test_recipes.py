"""Tests recipe selection, currency conversion eligibility, and role resolution."""

from __future__ import annotations

import contextvars
import json
import logging
import threading
import time
from collections.abc import Callable

import httpx
import pytest

from dc_graph import GraphClient, GraphReadError
from dfo_platform import Cause
from materialise import recipes as recipes_module
from materialise.errors import UpstreamResponseError
from materialise.graph import SubjectPage
from materialise.recipe_bodies import RECIPE_BODIES
from materialise.recipes import (
    CONVERSION_NODE_PROPERTIES,
    ConversionRecipe,
    EntityCurrencies,
    NodeArcs,
    NonCurrencyUnitError,
    RateCoverageGapError,
    RecipeNotDeclaredError,
    RoleAxisConflictError,
    RoleAxisNotDeclaredError,
    cause_from_recipe_error,
    check_nominal,
    check_nominal_natively_satisfied,
    currency_unmapped_error,
    graph_role_for_spec_role,
    read_declared_roles,
    require_currency_unit,
    resolve_convertible_recipe,
    resolve_deflator_role,
    resolve_entity_currencies,
    resolve_recipe,
)

from ._dup_arcs_fixtures import doubled_wire_arcs
from .conftest import FakeGraphReader

_DAC_PROVENANCE = "provenance/OECDCreditorReportingSystem"
_DAC_RECIPE = "ConversionRecipeDAC"
_WEO_RECIPE = "ConversionRecipeIMF"
_WEO_PROVENANCE = "provenance/IMFWorldEconomicOutlook"
_IDS_RECIPE = "ConversionRecipeIDS"
_IDS_PROVENANCE = "provenance/WorldBankInternationalDebtStatistics"
_DAC_DATASET = "dataset/OECDDAC"
_DAC_FALLBACK_PROVENANCE = "provenance/DAC1DatasetOnly"


def _node(**properties: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """One fixture node's arcs, typed as plain `tuple[str, ...]` values so
    a mix of fixed-length literal tuples (`("a",)`) and empty ones (`()`)
    across call sites doesn't widen the surrounding dict's inferred value
    type into a union `ty` then refuses to assign to `FakeGraphReader`'s
    own `Mapping[str, Mapping[str, tuple[str, ...]]]` parameter.
    """
    return properties


_CURRENCY_ARCS: dict[str, dict[str, tuple[str, ...]]] = {
    "CurrencyUSD": _node(typeOf=("CurrencyUnit",), nationalCurrencyOf=("country/USA",)),
    "CurrencyEUR": _node(
        typeOf=("CurrencyUnit",), nationalCurrencyOf=("country/FRA", "country/DEU")
    ),
    "CurrencyKES": _node(typeOf=("CurrencyUnit",), nationalCurrencyOf=("country/KEN",)),
    "CurrencyFRF": _node(typeOf=("CurrencyUnit",)),
    "NotACurrency": _node(typeOf=("StatisticalVariable",), nationalCurrencyOf=("country/ATL",)),
}


def _base_arcs() -> dict[str, dict[str, tuple[str, ...]]]:
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        _DAC_PROVENANCE: _node(conversionRecipe=(_DAC_RECIPE,), isPartOf=()),
        _WEO_PROVENANCE: _node(conversionRecipe=(_WEO_RECIPE,), isPartOf=()),
        _IDS_PROVENANCE: _node(conversionRecipe=(_IDS_RECIPE,), isPartOf=()),
        _DAC_FALLBACK_PROVENANCE: _node(conversionRecipe=(), isPartOf=(_DAC_DATASET,)),
        "provenance/Orphan": _node(conversionRecipe=(), isPartOf=()),
        _DAC_DATASET: _node(conversionRecipe=(_DAC_RECIPE,)),
        **_CURRENCY_ARCS,
        "CRS_Nominal_Flow": _node(
            observationProperties=("financingProvider", "financingRecipient"),
            measurementQualifier=("Nominal",),
            populationType=("FinancialFlow",),
        ),
        "StatVar_NoQualifier": _node(
            observationProperties=("financingProvider", "financingRecipient"),
            measurementQualifier=(),
            populationType=("FinancialFlow",),
        ),
        "StatVar_Stock": _node(
            observationProperties=("financingRecipient",),
            measurementQualifier=("Nominal",),
            populationType=("FinancialPosition",),
        ),
        "StatVar_NoPopulationType": _node(
            observationProperties=("financingRecipient",), measurementQualifier=("Nominal",)
        ),
        "WEO_Nominal_GDP": _node(
            measurementQualifier=("Nominal",), populationType=("EconomicProduction",)
        ),
        "StatVar_RealValue": _node(
            observationProperties=("financingProvider", "financingRecipient"),
            measurementQualifier=("RealValue",),
        ),
        "StatVar_ZeroRoleMatch": _node(
            observationProperties=("financingRecipient",),
            measurementQualifier=("Nominal",),
        ),
        "StatVar_NoObservationPropertiesArc": _node(measurementQualifier=("Nominal",)),
        "StatVar_EmptyObservationPropertiesArc": _node(
            observationProperties=(), measurementQualifier=("Nominal",)
        ),
    }
    return arcs


def _fake_graph(
    arcs: dict[str, dict[str, tuple[str, ...]]] | None = None,
) -> FakeGraphReader:
    return FakeGraphReader(arcs=arcs if arcs is not None else _base_arcs())


def _read(dcid: str, arcs: dict[str, dict[str, tuple[str, ...]]] | None = None) -> NodeArcs:
    """`dcid`'s arcs as a conversion decision reads them: one node read with
    `CONVERSION_NODE_PROPERTIES`, so a check reading any other property sees nothing."""
    return _fake_graph(arcs).node_arcs_many([dcid], properties=CONVERSION_NODE_PROPERTIES)[dcid]


def _resolve_recipe(
    provenance_id: str,
    arcs: dict[str, dict[str, tuple[str, ...]]] | None = None,
    *,
    graph: FakeGraphReader | GraphClient | None = None,
) -> ConversionRecipe:
    """`resolve_recipe` over the provenance arcs a decision read, with `graph` (by default
    a fake over `arcs`) serving the dataset fallback."""
    reader = graph if graph is not None else _fake_graph(arcs)
    provenance_arcs = reader.node_arcs_many([provenance_id], properties=CONVERSION_NODE_PROPERTIES)[
        provenance_id
    ]
    return resolve_recipe(reader, provenance_id=provenance_id, provenance_arcs=provenance_arcs)


def _resolve_convertible_recipe(*, stat_var: str, provenance_id: str) -> ConversionRecipe:
    graph = _fake_graph()
    arcs = graph.node_arcs_many([stat_var, provenance_id], properties=CONVERSION_NODE_PROPERTIES)
    return resolve_convertible_recipe(
        graph,
        stat_var=stat_var,
        stat_var_arcs=arcs[stat_var],
        provenance_id=provenance_id,
        provenance_arcs=arcs[provenance_id],
    )


# --- resolve_recipe ----------------------------------------------------


def test_resolve_recipe_returns_the_packages_own_dac_body() -> None:
    recipe = _resolve_recipe(_DAC_PROVENANCE)
    assert recipe is RECIPE_BODIES[_DAC_RECIPE]
    assert recipe.exchange_rate_variable == "ExchangeRate_CurrencyUnit_ConversionRecipeDAC"
    assert recipe.deflator_variable == "Deflator_EconomicProduction_ConversionRecipeDAC"
    assert recipe.deflator_entity_roles == frozenset({"financingProvider", "observationAbout"})
    assert recipe.flow_population_types == frozenset({"FinancialFlow", "FinancialObligation"})
    assert recipe.fallback_entity == "DACCountriesTotal"
    assert recipe.forecast_policy == "FlagProjected"
    assert recipe.table_provenance == "provenance/ONEConversionTablesDAC"


def test_resolve_recipe_returns_the_packages_own_imf_body() -> None:
    recipe = _resolve_recipe(_WEO_PROVENANCE)
    assert recipe is RECIPE_BODIES[_WEO_RECIPE]
    assert recipe.exchange_rate_variable == "ExchangeRate_CurrencyUnit_ConversionRecipeIMF"
    assert recipe.deflator_variable == "Deflator_EconomicProduction_ConversionRecipeIMF"
    assert recipe.deflator_entity_roles == frozenset({"financingProvider", "observationAbout"})
    assert recipe.flow_population_types == frozenset({"EconomicProduction"})
    assert recipe.fallback_entity is None
    assert recipe.forecast_policy == "FlagProjected"
    assert recipe.table_provenance == "provenance/ONEConversionTablesIMF"


def test_resolve_recipe_returns_the_ids_body_reading_the_imf_tables_by_the_debtor() -> None:
    recipe = _resolve_recipe(_IDS_PROVENANCE)
    imf = RECIPE_BODIES[_WEO_RECIPE]
    assert recipe is RECIPE_BODIES[_IDS_RECIPE]
    assert recipe.exchange_rate_variable == imf.exchange_rate_variable
    assert recipe.deflator_variable == imf.deflator_variable
    assert recipe.table_provenance == imf.table_provenance
    assert recipe.deflator_entity_roles == frozenset({"financingRecipient"})
    assert recipe.flow_population_types == frozenset({"FinancialFlow", "FinancialObligation"})
    assert recipe.fallback_entity is None


def test_resolve_recipe_falls_back_to_the_dataset() -> None:
    recipe = _resolve_recipe(_DAC_FALLBACK_PROVENANCE)
    assert recipe.recipe_id == _DAC_RECIPE


def test_resolve_recipe_refuses_recipe_undeclared_with_no_provenance_or_dataset_recipe() -> None:
    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        _resolve_recipe("provenance/Orphan")
    assert exc_info.value.code == "recipe_not_declared"
    assert exc_info.value.cause == "recipe_undeclared"


def test_resolve_recipe_refuses_recipe_undeclared_for_an_unheld_provenance() -> None:
    with pytest.raises(RecipeNotDeclaredError):
        _resolve_recipe("provenance/Nowhere")


def test_resolve_recipe_refuses_two_recipes_on_one_provenance() -> None:
    arcs = _base_arcs()
    arcs[_DAC_PROVENANCE] = {"conversionRecipe": (_DAC_RECIPE, _WEO_RECIPE), "isPartOf": ()}
    with pytest.raises(UpstreamResponseError):
        _resolve_recipe(_DAC_PROVENANCE, arcs)


def test_resolve_recipe_refuses_an_id_the_package_does_not_know() -> None:
    arcs = _base_arcs()
    arcs[_DAC_PROVENANCE] = {"conversionRecipe": ("recipe/Ghost",), "isPartOf": ()}
    with pytest.raises(UpstreamResponseError, match="recipe/Ghost"):
        _resolve_recipe(_DAC_PROVENANCE, arcs)


def test_resolve_recipe_ignores_body_arcs_a_fake_graph_also_carries() -> None:
    # The package recipe body wins when the graph node carries conflicting arcs.
    arcs = _base_arcs()
    arcs[_DAC_RECIPE] = {
        "exchangeRateVariable": ("Tampered_ExchangeRate",),
        "deflatorVariable": ("Tampered_Deflator",),
    }
    recipe = _resolve_recipe(_DAC_PROVENANCE, arcs)
    assert recipe is RECIPE_BODIES[_DAC_RECIPE]
    assert recipe.exchange_rate_variable == "ExchangeRate_CurrencyUnit_ConversionRecipeDAC"


def test_resolve_recipe_reads_the_graph_only_for_the_dataset_fallback() -> None:
    """A provenance declaring its recipe resolves from the arcs already read. One that
    declares none reads its Dataset's `conversionRecipe` alone."""
    graph = _fake_graph()
    provenance_arcs = _read(_DAC_PROVENANCE)
    resolve_recipe(graph, provenance_id=_DAC_PROVENANCE, provenance_arcs=provenance_arcs)
    assert graph.node_arcs_many_calls == []

    fallback_arcs = _read(_DAC_FALLBACK_PROVENANCE)
    resolve_recipe(graph, provenance_id=_DAC_FALLBACK_PROVENANCE, provenance_arcs=fallback_arcs)
    assert graph.node_arcs_many_calls == [((_DAC_DATASET,), ("conversionRecipe",))]


def _wire_graph(recipe_entries: list[dict[str, str]]) -> GraphClient:
    """A real `GraphClient` whose `/v2/node` answers the CRS provenance with
    `recipe_entries` as its `conversionRecipe` arc and no `isPartOf`."""
    body = json.dumps(
        {"data": {_DAC_PROVENANCE: {"arcs": {"conversionRecipe": {"nodes": recipe_entries}}}}}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "application/json"})

    return GraphClient(
        base_url="https://graph.example.org/core/api", transport=httpx.MockTransport(handler)
    )


def test_resolve_recipe_reads_an_identical_arc_asserted_twice_as_one_recipe() -> None:
    """Two imports asserting the same `conversionRecipe` arc come back as one
    `/v2/node` entry per provenance. The reader collapses them to one value."""
    arcs = doubled_wire_arcs(
        {"conversionRecipe": [_DAC_RECIPE]},
        provenances=("dc/base/one_oecd_crs", "dc/base/one_conversion_recipe_patch"),
    )
    graph = _wire_graph(arcs["conversionRecipe"]["nodes"])
    assert _resolve_recipe(_DAC_PROVENANCE, graph=graph) is RECIPE_BODIES[_DAC_RECIPE]


def test_resolve_recipe_refuses_two_different_recipes_asserted_by_two_imports() -> None:
    graph = _wire_graph(
        [
            {"dcid": _DAC_RECIPE, "provenanceId": "dc/base/one_oecd_crs"},
            {"dcid": _WEO_RECIPE, "provenanceId": "dc/base/one_conversion_recipe_patch"},
        ]
    )
    with pytest.raises(UpstreamResponseError, match="more than one conversionRecipe"):
        _resolve_recipe(_DAC_PROVENANCE, graph=graph)


# --- resolve_convertible_recipe -----------------------------------------


def test_resolve_convertible_recipe_happy_path() -> None:
    recipe = _resolve_convertible_recipe(stat_var="CRS_Nominal_Flow", provenance_id=_DAC_PROVENANCE)
    assert recipe.recipe_id == _DAC_RECIPE


def test_check_nominal_passes_for_a_nominal_statvar() -> None:
    check_nominal(_read("CRS_Nominal_Flow"), stat_var="CRS_Nominal_Flow")


def test_check_nominal_refuses_a_statvar_with_no_qualifier() -> None:
    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        check_nominal(_read("StatVar_NoQualifier"), stat_var="StatVar_NoQualifier")
    assert exc_info.value.cause == "nominal_undeclared"


def test_check_nominal_refuses_a_non_nominal_qualifier() -> None:
    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        check_nominal(_read("StatVar_RealValue"), stat_var="StatVar_RealValue")
    assert exc_info.value.cause == "nominal_undeclared"


def test_check_nominal_natively_satisfied_passes_for_a_nominal_statvar() -> None:
    check_nominal_natively_satisfied(_read("CRS_Nominal_Flow"), stat_var="CRS_Nominal_Flow")


def test_check_nominal_natively_satisfied_passes_with_no_qualifier_arc() -> None:
    # Native satisfaction skips conversion checks and accepts a StatVar without this arc.
    check_nominal_natively_satisfied(_read("StatVar_NoQualifier"), stat_var="StatVar_NoQualifier")


def test_check_nominal_natively_satisfied_refuses_a_non_nominal_qualifier() -> None:
    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        check_nominal_natively_satisfied(_read("StatVar_RealValue"), stat_var="StatVar_RealValue")
    assert exc_info.value.cause == "nominal_undeclared"


def test_resolve_convertible_recipe_refuses_a_statvar_with_no_qualifier() -> None:
    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        _resolve_convertible_recipe(stat_var="StatVar_NoQualifier", provenance_id=_DAC_PROVENANCE)
    assert exc_info.value.cause == "nominal_undeclared"


def test_resolve_convertible_recipe_refuses_a_non_nominal_qualifier() -> None:
    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        _resolve_convertible_recipe(stat_var="StatVar_RealValue", provenance_id=_DAC_PROVENANCE)
    assert exc_info.value.cause == "nominal_undeclared"


def test_resolve_convertible_recipe_refuses_when_no_recipe_is_declared() -> None:
    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        _resolve_convertible_recipe(stat_var="CRS_Nominal_Flow", provenance_id="provenance/Orphan")
    assert exc_info.value.cause == "recipe_undeclared"


@pytest.mark.parametrize("stat_var", ["StatVar_Stock", "StatVar_NoPopulationType"])
def test_resolve_convertible_recipe_refuses_a_population_type_outside_the_flow_set(
    stat_var: str,
) -> None:
    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        _resolve_convertible_recipe(stat_var=stat_var, provenance_id=_IDS_PROVENANCE)
    assert exc_info.value.cause == "stock_unsupported"


def test_resolve_convertible_recipe_reads_the_flow_set_of_the_resolved_recipe() -> None:
    """WEO's `EconomicProduction` converts under the IMF recipe and refuses
    under the DAC recipe, whose set lists only financial flows."""
    recipe = _resolve_convertible_recipe(stat_var="WEO_Nominal_GDP", provenance_id=_WEO_PROVENANCE)
    assert recipe.recipe_id == _WEO_RECIPE
    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        _resolve_convertible_recipe(stat_var="WEO_Nominal_GDP", provenance_id=_DAC_PROVENANCE)
    assert exc_info.value.cause == "stock_unsupported"


# --- role resolution -----------------------------------------------------


def test_read_declared_roles() -> None:
    assert read_declared_roles(_fake_graph(), "CRS_Nominal_Flow") == (
        "financingProvider",
        "financingRecipient",
    )


def test_read_declared_roles_defaults_to_the_subject_role_for_an_absent_arc() -> None:
    assert read_declared_roles(_fake_graph(), "StatVar_NoObservationPropertiesArc") == (
        "observationAbout",
    )


def test_read_declared_roles_defaults_to_the_subject_role_for_an_empty_arc() -> None:
    assert read_declared_roles(_fake_graph(), "StatVar_EmptyObservationPropertiesArc") == (
        "observationAbout",
    )


def test_resolve_deflator_role_resolves_the_subject_role_against_a_dac_shaped_set() -> None:
    recipe = _resolve_recipe(_DAC_PROVENANCE)
    role = resolve_deflator_role(recipe, declared_roles=("observationAbout",))
    assert role == "observationAbout"


def test_resolve_deflator_role_resolves_the_subject_role_against_a_weo_shaped_set() -> None:
    recipe = _resolve_recipe(_WEO_PROVENANCE)
    role = resolve_deflator_role(recipe, declared_roles=("observationAbout",))
    assert role == "observationAbout"


@pytest.mark.parametrize(
    "declared_roles", [("financingRecipient",), ("financingProvider", "financingRecipient")]
)
def test_resolve_deflator_role_picks_the_debtor_under_the_ids_recipe(
    declared_roles: tuple[str, ...],
) -> None:
    """IDS StatVars declare the debtor alone or creditor by debtor. Both
    deflate by the debtor."""
    recipe = _resolve_recipe(_IDS_PROVENANCE)
    assert resolve_deflator_role(recipe, declared_roles=declared_roles) == "financingRecipient"


def test_resolve_deflator_role_matches_the_one_declared_role_in_the_set() -> None:
    recipe = _resolve_recipe(_DAC_PROVENANCE)
    role = resolve_deflator_role(recipe, declared_roles=("financingProvider", "financingRecipient"))
    assert role == "financingProvider"


def test_resolve_deflator_role_refuses_zero_matches() -> None:
    recipe = _resolve_recipe(_DAC_PROVENANCE)
    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        resolve_deflator_role(recipe, declared_roles=("financingRecipient",))
    assert exc_info.value.cause == "role_undeclared"


def test_resolve_deflator_role_refuses_more_than_one_match() -> None:
    recipe = ConversionRecipe(
        recipe_id="recipe/TwoAxisTest",
        exchange_rate_variable="ExchangeRate_X",
        deflator_variable="Deflator_X",
        deflator_entity_roles=frozenset({"financingProvider", "financingRecipient"}),
        flow_population_types=frozenset({"FinancialFlow"}),
        forecast_policy="FlagProjected",
        table_provenance="provenance/Test",
    )
    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        resolve_deflator_role(recipe, declared_roles=("financingProvider", "financingRecipient"))
    assert exc_info.value.cause == "role_undeclared"


def test_graph_role_for_spec_role_origin_and_destination() -> None:
    declared = ("financingProvider", "financingRecipient")
    assert graph_role_for_spec_role(spec_role="origin", declared_roles=declared) == (
        "financingProvider"
    )
    assert graph_role_for_spec_role(spec_role="destination", declared_roles=declared) == (
        "financingRecipient"
    )


def test_graph_role_for_spec_role_observation_about() -> None:
    assert (
        graph_role_for_spec_role(spec_role="observationAbout", declared_roles=("observationAbout",))
        == "observationAbout"
    )


def test_graph_role_for_spec_role_refuses_an_undeclared_axis() -> None:
    with pytest.raises(RoleAxisNotDeclaredError, match="destination") as exc_info:
        graph_role_for_spec_role(spec_role="destination", declared_roles=("financingProvider",))
    assert exc_info.value.cause == "role_undeclared"


def test_graph_role_for_spec_role_refuses_undeclared_observation_about() -> None:
    with pytest.raises(RoleAxisNotDeclaredError, match="observationAbout") as exc_info:
        graph_role_for_spec_role(
            spec_role="observationAbout", declared_roles=("financingProvider",)
        )
    assert exc_info.value.cause == "role_undeclared"


def test_graph_role_for_spec_role_refuses_a_mixed_collection_subject_only_member() -> None:
    """A collection mixing a bilateral-flow StatVar (origin/destination)
    with a subject-only one (observationAbout only) under a spec asking for
    `destination` passes the spec's own whole-spec role-key validation, then
    reaches this function for the subject-only member.
    """
    with pytest.raises(RoleAxisNotDeclaredError) as exc_info:
        graph_role_for_spec_role(spec_role="destination", declared_roles=("observationAbout",))
    assert exc_info.value.cause == "role_undeclared"


def test_role_axis_not_declared_error_is_a_recipe_not_declared_error() -> None:
    """A `RecipeNotDeclaredError` subclass, not an independent class: a
    caller such as `datasets_api.adapter.to_materialisation_request`, which
    catches the base class broadly, must still catch this one.
    """
    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        graph_role_for_spec_role(spec_role="destination", declared_roles=("observationAbout",))
    assert isinstance(exc_info.value, RoleAxisNotDeclaredError)
    assert exc_info.value.code == "recipe_not_declared"
    assert exc_info.value.status == 422


def test_graph_role_for_spec_role_refuses_two_declared_roles_on_one_axis() -> None:
    """`materialise.roles.roles_by_axis` raises a bare `ValueError` for a
    shape whose two declared roles name the same flow axis. This function
    translates it into `RoleAxisConflictError` so a caller's
    `except RecipeNotDeclaredError` catches it.
    """
    with pytest.raises(RoleAxisConflictError, match="origin") as exc_info:
        graph_role_for_spec_role(spec_role="origin", declared_roles=("provider", "creditor"))
    message = str(exc_info.value)
    assert "provider" in message
    assert "creditor" in message
    assert exc_info.value.cause == "role_undeclared"
    assert exc_info.value.code == "recipe_not_declared"
    assert exc_info.value.status == 422


def test_role_axis_conflict_error_is_a_recipe_not_declared_error() -> None:
    """`RoleAxisConflictError` is a `RecipeNotDeclaredError` subclass, so the
    same adapter's `except RecipeNotDeclaredError` catch sees the
    two-roles-on-one-axis case too, instead of it reaching the mint door's
    generic handler as a 500.
    """
    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        graph_role_for_spec_role(spec_role="origin", declared_roles=("provider", "creditor"))
    assert isinstance(exc_info.value, RoleAxisConflictError)
    assert exc_info.value.code == "recipe_not_declared"
    assert exc_info.value.status == 422


def test_graph_role_for_spec_role_refuses_a_spec_role_outside_the_closed_vocabulary() -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        graph_role_for_spec_role(
            spec_role="notARealSpecRole",  # type: ignore[arg-type]
            declared_roles=("financingProvider",),
        )


# --- currency resolution ---------------------------------------------------


@pytest.mark.parametrize("currency", ["CurrencyUSD", "CurrencyFRF"])
def test_require_currency_unit_accepts_a_currency_unit_node(currency: str) -> None:
    """A withdrawn currency (`CurrencyFRF`) is still a `CurrencyUnit`. Its missing rate
    rows refuse later, at conversion."""
    assert require_currency_unit(_read(currency), currency=currency) == currency


@pytest.mark.parametrize("currency", ["CurrencyXXX", "NotACurrency"])
def test_require_currency_unit_refuses_an_unheld_or_non_currency_node(currency: str) -> None:
    with pytest.raises(NonCurrencyUnitError) as exc_info:
        require_currency_unit(_read(currency), currency=currency)
    assert exc_info.value.code == "facet_binding_unsupported"
    assert exc_info.value.unit == currency


def test_resolve_entity_currencies_reads_nationalcurrencyof_in_reverse() -> None:
    graph = _fake_graph()
    assert resolve_entity_currencies(
        graph, entities=["country/KEN", "country/DEU"]
    ) == EntityCurrencies(
        currency_by_entity={"country/KEN": "CurrencyKES", "country/DEU": "CurrencyEUR"},
        unmapped=(),
    )
    assert graph.calls["node_subjects"] == 2
    # One node read types both currencies.
    assert graph.node_arcs_many_calls == [(("CurrencyKES", "CurrencyEUR"), ("typeOf",))]


def _two_currency_graph() -> FakeGraphReader:
    """Lesotho named by two currencies. An ISO ask for either currency still works."""
    return _fake_graph(
        arcs={
            **_base_arcs(),
            "CurrencyZAR": _node(typeOf=("CurrencyUnit",), nationalCurrencyOf=("country/LSO",)),
            "CurrencyLSL": _node(typeOf=("CurrencyUnit",), nationalCurrencyOf=("country/LSO",)),
        }
    )


def _truncated_page_graph() -> FakeGraphReader:
    return FakeGraphReader(
        arcs=_base_arcs(),
        subjects={
            ("country/LSO", "nationalCurrencyOf"): SubjectPage(
                dcids=("CurrencyLSL",), truncated=True
            )
        },
    )


@pytest.mark.parametrize(
    ("graph", "unmapped", "logged"),
    [
        pytest.param(_fake_graph, "EuropeanUnionInstitutions", [], id="named_by_no_currency"),
        pytest.param(
            _two_currency_graph,
            "country/LSO",
            [("country/LSO", ["CurrencyLSL", "CurrencyZAR"], False)],
            id="named_by_two_currencies",
        ),
        pytest.param(
            _truncated_page_graph,
            "country/LSO",
            [("country/LSO", ["CurrencyLSL"], True)],
            id="truncated_reverse_read",
        ),
    ],
)
def test_resolve_entity_currencies_returns_the_unmapped_entities_in_input_order(
    graph: Callable[[], FakeGraphReader],
    unmapped: str,
    logged: list[tuple[str, list[str], bool]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An entity no single currency names is returned unmapped, in input order, beside the
    mapped ones; the caller decides whether its rows are omitted or the request refuses. One
    several currencies name, or whose read was truncated, breaks the one-currency rule, and
    a warning records the entity, the currencies read and the truncation."""
    with caplog.at_level(logging.WARNING, logger=recipes_module.__name__):
        currencies = resolve_entity_currencies(graph(), entities=[unmapped, "country/KEN"])
    assert currencies == EntityCurrencies(
        currency_by_entity={"country/KEN": "CurrencyKES"}, unmapped=(unmapped,)
    )
    warnings = [r for r in caplog.records if r.name == recipes_module.__name__]
    assert [r.levelno for r in warnings] == [logging.WARNING] * len(logged)
    assert [
        (entity, sorted(dcids), truncated)
        for entity, dcids, truncated in (r.args for r in warnings)
    ] == logged


@pytest.mark.parametrize(
    ("entities", "named", "truncated"),
    [
        (
            ["southOfSahara", "africaRegion", "africaRegion"],
            ("africaRegion", "southOfSahara"),
            False,
        ),
        (
            [f"region/R{i:02d}" for i in reversed(range(12))],
            tuple(f"region/R{i:02d}" for i in range(10)),
            True,
        ),
    ],
)
def test_currency_unmapped_error_names_the_entities_in_dcid_order(
    entities: list[str], named: tuple[str, ...], truncated: bool
) -> None:
    error = currency_unmapped_error(entities)
    assert (error.code, error.cause) == ("rate_coverage_gap", "currency_unmapped")
    assert (error.entities, error.entities_truncated) == (named, truncated)


def test_resolve_entity_currencies_refuses_a_subject_that_is_not_a_currency_unit() -> None:
    with pytest.raises(NonCurrencyUnitError):
        resolve_entity_currencies(_fake_graph(), entities=["country/ATL", "country/KEN"])


class _SlowReverseReads(FakeGraphReader):
    """A graph whose reverse reads each take `delay` seconds, recording the most in flight
    at once."""

    def __init__(self, *, delay: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self._delay = delay
        self._lock = threading.Lock()
        self._in_flight = 0
        self.max_in_flight = 0

    def node_subjects(self, dcid: str, **kwargs) -> SubjectPage:
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            time.sleep(self._delay)
            return super().node_subjects(dcid, **kwargs)
        finally:
            with self._lock:
                self._in_flight -= 1


def test_resolve_entity_currencies_reads_many_entities_concurrently_within_a_bound() -> None:
    """150 recipients' reverse reads run at most `_ENTITY_CURRENCY_READS_MAX` at a time, so
    the serial depth is the entity count over the pool width, and one node read types every
    currency."""
    entities = [f"country/C{i:03d}" for i in range(150)]
    arcs = {
        **_base_arcs(),
        **{
            f"Currency{i:03d}": _node(typeOf=("CurrencyUnit",), nationalCurrencyOf=(entity,))
            for i, entity in enumerate(entities)
        },
    }
    graph = _SlowReverseReads(delay=0.02, arcs=arcs)
    started = time.monotonic()
    currencies = resolve_entity_currencies(graph, entities=entities)
    elapsed = time.monotonic() - started
    assert currencies.currency_by_entity == {
        entity: f"Currency{i:03d}" for i, entity in enumerate(entities)
    }
    assert graph.calls["node_subjects"] == 150
    assert graph.calls["node_arcs_many"] == 1
    assert graph.max_in_flight == recipes_module._ENTITY_CURRENCY_READS_MAX
    # Serially the reads take 3 s; ten rounds of 16 take about 0.2 s.
    assert elapsed < 1.5


def test_resolve_entity_currencies_reads_under_the_callers_context() -> None:
    """Each pooled read runs under a copy of the caller's context, so request-scoped state
    such as the bound origin catalogue reaches it."""
    marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker", default="unset")
    seen: list[str] = []

    class _Seeing(FakeGraphReader):
        def node_subjects(self, dcid: str, **kwargs) -> SubjectPage:
            seen.append(marker.get())
            return super().node_subjects(dcid, **kwargs)

    token = marker.set("request")
    try:
        resolve_entity_currencies(
            _Seeing(arcs=_base_arcs()), entities=["country/KEN", "country/DEU"]
        )
    finally:
        marker.reset(token)
    assert seen == ["request", "request"]


class _OneReverseReadFails(FakeGraphReader):
    """A graph whose reverse read of `failing` raises `GraphReadError` once a full pool of
    reads is in flight; every other read takes `delay` seconds. Records which entities were
    read and how many reads had finished when the error was raised."""

    def __init__(self, *, failing: str, in_flight: int, delay: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self._failing = failing
        self._in_flight = in_flight
        self._delay = delay
        self._condition = threading.Condition()
        self.started: list[str] = []
        self.finished = 0

    def node_subjects(self, dcid: str, **kwargs) -> SubjectPage:
        with self._condition:
            self.started.append(dcid)
            self._condition.notify_all()
            if dcid == self._failing:
                self._condition.wait_for(lambda: len(self.started) >= self._in_flight, timeout=5)
                raise GraphReadError(f"reverse read of {dcid} failed", upstream_status=503)
        time.sleep(self._delay)
        page = super().node_subjects(dcid, **kwargs)
        with self._condition:
            self.finished += 1
        return page


def test_resolve_entity_currencies_raises_when_one_pooled_read_fails() -> None:
    """One of 40 reverse reads fails while a full pool of others runs. The call raises that
    error with no partial map, waits for the running reads, and cancels the queued ones, so
    they are never requested."""
    width = recipes_module._ENTITY_CURRENCY_READS_MAX
    entities = [f"country/C{i:03d}" for i in range(40)]
    arcs = {
        **_base_arcs(),
        **{
            f"Currency{i:03d}": _node(typeOf=("CurrencyUnit",), nationalCurrencyOf=(entity,))
            for i, entity in enumerate(entities)
        },
    }
    graph = _OneReverseReadFails(failing=entities[0], in_flight=width, delay=0.1, arcs=arcs)

    with pytest.raises(GraphReadError, match="country/C000"):
        resolve_entity_currencies(graph, entities=entities)

    # The worker freed by the failure can take one queued read before the cancel lands.
    assert width <= len(graph.started) <= width + 1
    assert set(graph.started[:width]) == set(entities[:width])
    # Every started read other than the failed one finished before the call returned.
    assert graph.finished == len(graph.started) - 1
    assert graph.calls["node_arcs_many"] == 0


# --- RateCoverageGapError truncation --------------------------------------


def test_rate_coverage_gap_error_caps_named_entities() -> None:
    entities = tuple(f"entity/{i}" for i in range(15))
    error = RateCoverageGapError("gap", cause="currency_unmapped", entities=entities)
    assert error.entities == entities[:10]
    assert error.entities_truncated is True


def test_rate_coverage_gap_error_does_not_truncate_a_short_list() -> None:
    error = RateCoverageGapError("gap", cause="currency_unmapped", entities=("a", "b"))
    assert error.entities == ("a", "b")
    assert error.entities_truncated is False


# --- cause_from_recipe_error -----------------------------------------------


def test_cause_from_recipe_error_recipe_not_declared_carries_no_entities() -> None:
    error = RecipeNotDeclaredError("no recipe", cause="recipe_undeclared")
    cause = cause_from_recipe_error(error)
    assert cause == Cause(code="recipe_undeclared", entities=[], truncated=False)


def test_cause_from_recipe_error_rate_coverage_gap_carries_its_entities() -> None:
    error = RateCoverageGapError(
        "gap", cause="rate_missing", entities=["EntityA@2020", "EntityB@2021"]
    )
    cause = cause_from_recipe_error(error)
    assert cause == Cause(
        code="rate_missing", entities=["EntityA@2020", "EntityB@2021"], truncated=False
    )


def test_cause_from_recipe_error_rate_coverage_gap_truncation_carries_through() -> None:
    entities = [f"Entity{i}@2020" for i in range(15)]
    error = RateCoverageGapError("gap", cause="deflator_missing", entities=entities)
    cause = cause_from_recipe_error(error)
    assert cause.entities == entities[:10]
    assert cause.truncated is True
