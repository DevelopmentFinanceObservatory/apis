"""Tests member conversion planning and application with graph and SDMX fixtures."""

from __future__ import annotations

import pyarrow as pa
import pytest

from dc_sdmx import SdmxClient
from materialise import member_conversion
from materialise.convert import merged_record
from materialise.errors import MaterialisationUsageError
from materialise.facets import FacetBinding, FacetSelection, UnitRequest
from materialise.generation import _convert_by_series, _split_by_roles
from materialise.graph import Facet
from materialise.member_conversion import (
    MemberConversionPlan,
    apply_member_conversion,
    apply_member_conversion_by_series,
    coverage_decision,
    omit_unmapped_rows,
    plan_member_conversion,
    require_decision_coverage,
    served_block,
)
from materialise.pipeline import MaterialisationRequest, decide_member_unit
from materialise.rate_tables import RateTableCache
from materialise.recipe_bodies import RECIPE_BODIES
from materialise.recipes import (
    _MAX_NAMED_ENTITIES,
    NonCurrencyUnitError,
    RateCoverageGapError,
    RecipeNotDeclaredError,
)
from materialise.series_conversion import SeriesConversion, series_key
from materialise.window import RequestedWindow
from tests.engine._fake_coverage import build_coverage_index

from ._rate_table_support import (
    RATE_CSV_HEADER,
    RateTableServer,
    query_params,
    rate_row,
    recipe_table_bodies,
)
from ._table_support import table_from_cells
from .conftest import FakeGraphReader

# Subject-only precomputed candidates bypass this required coverage index.
_UNUSED_COVERAGE = build_coverage_index([], role_names={})

_STAT_VAR = "Amount_FinancialFlow_DevelopmentAssistance_Nominal"
_ROLES = ("financingProvider", "financingRecipient")
_DAC_RECIPE = "ConversionRecipeDAC"
_MEMBER_PROVENANCE = "provenance/OECDCreditorReportingSystem"
_TABLE_PROVENANCE = RECIPE_BODIES[_DAC_RECIPE].table_provenance
_EXCHANGE_VAR = RECIPE_BODIES[_DAC_RECIPE].exchange_rate_variable
_DEFLATOR_VAR = RECIPE_BODIES[_DAC_RECIPE].deflator_variable


def _rate_row(*, stat_var: str, entity: str, year: str, value: str, facet_id: str = "f1") -> str:
    return rate_row(
        stat_var=stat_var,
        entity=entity,
        year=year,
        value=value,
        provenance=_TABLE_PROVENANCE,
        facet_id=facet_id,
    )


def _multi_sdmx(bodies_by_stat_var: dict[str, str]) -> tuple[SdmxClient, RateTableServer]:
    """One client serving each table StatVar's own rows over `/availability`
    and `/data`, so a constant-prices conversion's two loads stay apart the
    way the two real StatVars would.
    """
    server = RateTableServer(bodies_by_stat_var)
    return server.client(), server


def _base_arcs(**statvar_overrides: tuple[str, ...]) -> dict[str, dict[str, tuple[str, ...]]]:
    statvar_arcs: dict[str, tuple[str, ...]] = {
        "observationProperties": _ROLES,
        "measurementQualifier": ("Nominal",),
        "populationType": ("FinancialFlow",),
    }
    statvar_arcs.update(statvar_overrides)
    return {
        _MEMBER_PROVENANCE: {"conversionRecipe": (_DAC_RECIPE,), "isPartOf": ()},
        _TABLE_PROVENANCE: {"lastDataRefreshDate": ("2026-01-01T00:00:00Z",)},
        "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
        "CurrencyEUR": {"typeOf": ("CurrencyUnit",)},
        "CurrencyKES": {"typeOf": ("CurrencyUnit",)},
        "CurrencyGBP": {"typeOf": ("CurrencyUnit",)},
        "CurrencyFRF": {"typeOf": ("CurrencyUnit",)},
        _STAT_VAR: statvar_arcs,
    }


def _graph(
    arcs: dict[str, dict[str, tuple[str, ...]]] | None = None,
) -> FakeGraphReader:
    return FakeGraphReader(arcs=arcs if arcs is not None else _base_arcs())


def _with_entity_currency(entity: str, currency: str) -> dict[str, dict[str, tuple[str, ...]]]:
    arcs = _base_arcs()
    arcs[currency] = {**arcs[currency], "nationalCurrencyOf": (entity,)}
    return arcs


def _selection(axis: str, ref: str | None, *, request: UnitRequest | None = None) -> FacetSelection:
    return FacetSelection(
        axis=axis,
        ref=ref,
        basis="bound",
        default_policy=None,
        default_policy_version=None,
        selection_reason=None,
        dispatch_facet_id="f1",
        request=request,
    )


def _selections(
    *, native_unit: str, provenance: str = _MEMBER_PROVENANCE, request: UnitRequest | None
) -> dict[str, FacetSelection]:
    return {
        "unit": _selection("unit", native_unit, request=request),
        "provenance": _selection("provenance", provenance),
        "measurementMethod": _selection("measurementMethod", None),
    }


def _request(
    *,
    stat_var: str = _STAT_VAR,
    declared_roles: tuple[str, ...] = _ROLES,
    role_bindings: dict[str, list[str]] | None = None,
) -> MaterialisationRequest:
    bindings = (
        role_bindings
        if role_bindings is not None
        else {
            "financingProvider": ["country/FRA"],
            "financingRecipient": ["country/KEN"],
        }
    )
    return MaterialisationRequest(
        stat_var=stat_var,
        role_bindings=bindings,
        declared_roles=declared_roles,
        facet_bindings={},
        window=RequestedWindow(kind="all"),
    )


def _member_rows(records) -> pa.Table:
    return table_from_cells(records, roles=_ROLES)


def _convert(plan: MemberConversionPlan | None, rows: pa.Table):
    """`rows` converted as a read converts them: `omit_unmapped_rows` over every row, then
    `apply_member_conversion` into the currencies it resolved."""
    kept, currencies = omit_unmapped_rows(plan, rows, served=rows)
    return apply_member_conversion(plan, kept, declared_roles=_ROLES, currencies=currencies)


def _row(*, provider: str = "country/FRA", recipient: str = "country/KEN", year: str, value: str):
    return {
        "variableMeasured": _STAT_VAR,
        "financingProvider": provider,
        "financingRecipient": recipient,
        "TIME_PERIOD": year,
        "OBS_VALUE": value,
    }


def test_no_request_binding_returns_none_plan():
    selections = _selections(native_unit="CurrencyUSD", request=None)
    sdmx, _ = _multi_sdmx({})
    plan = plan_member_conversion(
        request=_request(),
        selections=selections,
        graph=_graph(),
        sdmx=sdmx,
        rate_cache=RateTableCache(),
    )
    assert plan is None


def test_apply_with_no_plan_matches_collapse_alone():
    from materialise.facet_collapse import collapse_duplicate_identity_facets_table

    rows = _member_rows([_row(year="2022", value="100")])
    table, record = apply_member_conversion(None, rows, declared_roles=_ROLES)
    assert table.equals(collapse_duplicate_identity_facets_table(rows, declared_roles=_ROLES))
    assert record is None


def test_native_usd_current_converts_nothing_and_writes_struct_cells():
    unit_request = UnitRequest(currency="USD", prices="current")
    selections = _selections(native_unit="CurrencyUSD", request=unit_request)
    sdmx, server = _multi_sdmx({})

    plan = plan_member_conversion(
        request=_request(),
        selections=selections,
        graph=_graph(),
        sdmx=sdmx,
        rate_cache=RateTableCache(),
    )
    assert plan is not None
    assert plan.natively_satisfied is True
    assert plan.recipe is None

    rows = _member_rows([_row(year="2022", value="100")])
    table, record = apply_member_conversion(plan, rows, declared_roles=_ROLES)

    assert server.requests == []  # no rate/deflator fetch at all
    assert record is None
    assert table["OBS_VALUE"].to_pylist() == ["100"]
    assert table["unit"].to_pylist() == [
        {"currency": "USD", "prices": "current", "base_year": None}
    ]


def test_native_satisfaction_plans_and_applies_with_no_recipe_declared():
    # Native satisfaction skips recipe resolution and succeeds under a provenance without a recipe.
    unit_request = UnitRequest(currency="USD", prices="current")
    selections = _selections(
        native_unit="CurrencyUSD", provenance="provenance/Orphan", request=unit_request
    )
    arcs = _base_arcs()
    arcs["provenance/Orphan"] = {"conversionRecipe": (), "isPartOf": ()}

    plan = plan_member_conversion(
        request=_request(),
        selections=selections,
        graph=_graph(arcs),
        sdmx=_multi_sdmx({})[0],
        rate_cache=RateTableCache(),
    )
    assert plan is not None
    assert plan.recipe is None

    rows = _member_rows([_row(year="2022", value="100")])
    table, record = apply_member_conversion(plan, rows, declared_roles=_ROLES)
    assert record is None
    assert table["unit"].to_pylist() == [
        {"currency": "USD", "prices": "current", "base_year": None}
    ]


def test_native_satisfaction_plans_with_no_measurement_qualifier_arc():
    # A native request needs no arithmetic or qualifier arc.
    unit_request = UnitRequest(currency="USD", prices="current")
    selections = _selections(native_unit="CurrencyUSD", request=unit_request)
    arcs = _base_arcs(measurementQualifier=())
    plan = plan_member_conversion(
        request=_request(),
        selections=selections,
        graph=_graph(arcs),
        sdmx=_multi_sdmx({})[0],
        rate_cache=RateTableCache(),
    )
    assert plan is not None
    assert plan.recipe is None


def test_native_satisfaction_refuses_a_declared_non_nominal_qualifier():
    unit_request = UnitRequest(currency="USD", prices="current")
    selections = _selections(native_unit="CurrencyUSD", request=unit_request)
    arcs = _base_arcs(measurementQualifier=("RealValue",))
    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        plan_member_conversion(
            request=_request(),
            selections=selections,
            graph=_graph(arcs),
            sdmx=_multi_sdmx({})[0],
            rate_cache=RateTableCache(),
        )
    assert exc_info.value.cause == "nominal_undeclared"


def test_native_satisfaction_keeps_struct_type_on_a_zero_row_table():
    unit_request = UnitRequest(currency="USD", prices="current")
    selections = _selections(native_unit="CurrencyUSD", request=unit_request)
    plan = plan_member_conversion(
        request=_request(),
        selections=selections,
        graph=_graph(),
        sdmx=_multi_sdmx({})[0],
        rate_cache=RateTableCache(),
    )
    assert plan is not None

    empty_rows = _member_rows([])
    table, record = apply_member_conversion(plan, empty_rows, declared_roles=_ROLES)

    assert table.num_rows == 0
    assert pa.types.is_struct(table.schema.field("unit").type)
    assert record is None


def test_eur_from_native_usd_loads_one_rate_table_and_converts():
    unit_request = UnitRequest(currency="EUR", prices="current")
    selections = _selections(native_unit="CurrencyUSD", request=unit_request)
    sdmx, server = _multi_sdmx(
        {
            _EXCHANGE_VAR: RATE_CSV_HEADER
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2022", value="1.0")
            + _rate_row(
                stat_var=_EXCHANGE_VAR,
                entity="CurrencyEUR",
                year="2022",
                value="0.9",
            )
        }
    )

    plan = plan_member_conversion(
        request=_request(),
        selections=selections,
        graph=_graph(),
        sdmx=sdmx,
        rate_cache=RateTableCache(),
    )
    assert plan is not None
    assert plan.uniform_target is not None
    assert len(server.data_requests()) == 1  # loaded eagerly at plan time, one exchange-rate fetch

    rows = _member_rows([_row(year="2022", value="100")])
    table, record = apply_member_conversion(plan, rows, declared_roles=_ROLES)

    assert len(server.data_requests()) == 1  # apply reuses the plan's own already-loaded table
    assert table["OBS_VALUE"].to_pylist() == [repr(100.0 * 0.9 / 1.0)]
    assert table["unit"].to_pylist() == [
        {"currency": "EUR", "prices": "current", "base_year": None}
    ]
    assert record is not None
    assert record.recipe_id == _DAC_RECIPE
    assert record.deflator_table_version is None


def test_a_us_dollar_native_unit_satisfies_a_usd_request_as_currency_usd():
    """`USDollar` resolves through its pinned ISO code to `CurrencyUSD`: no conversion runs."""
    unit_request = UnitRequest(currency="USD", prices="current")
    selections = _selections(native_unit="USDollar", request=unit_request)
    sdmx, server = _multi_sdmx({})

    plan = plan_member_conversion(
        request=_request(),
        selections=selections,
        graph=_graph(),
        sdmx=sdmx,
        rate_cache=RateTableCache(),
    )

    assert plan is not None
    assert plan.natively_satisfied is True
    assert plan.native_currency == "CurrencyUSD"
    assert server.requests == []


def test_eur_from_a_us_dollar_native_unit_converts_on_currency_usd_rates():
    unit_request = UnitRequest(currency="EUR", prices="current")
    selections = _selections(native_unit="USDollar", request=unit_request)
    sdmx, _ = _multi_sdmx(
        {
            _EXCHANGE_VAR: RATE_CSV_HEADER
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2022", value="1.0")
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyEUR", year="2022", value="0.9")
        }
    )

    plan = plan_member_conversion(
        request=_request(),
        selections=selections,
        graph=_graph(),
        sdmx=sdmx,
        rate_cache=RateTableCache(),
    )
    assert plan is not None
    assert plan.native_currency == "CurrencyUSD"

    rows = _member_rows([_row(year="2022", value="100")])
    table, record = apply_member_conversion(plan, rows, declared_roles=_ROLES)

    assert table["OBS_VALUE"].to_pylist() == [repr(100.0 * 0.9 / 1.0)]
    assert record is not None
    assert record.native_currency == "CurrencyUSD"


def test_constant_prices_load_both_tables():
    unit_request = UnitRequest(currency="USD", prices="constant", base_year=2015)
    selections = _selections(native_unit="CurrencyUSD", request=unit_request)
    sdmx, server = _multi_sdmx(
        {
            _EXCHANGE_VAR: RATE_CSV_HEADER
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2015", value="1.0")
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2020", value="1.0"),
            _DEFLATOR_VAR: RATE_CSV_HEADER
            + _rate_row(stat_var=_DEFLATOR_VAR, entity="country/FRA", year="2015", value="100.0")
            + _rate_row(stat_var=_DEFLATOR_VAR, entity="country/FRA", year="2020", value="110.0"),
        }
    )

    plan = plan_member_conversion(
        request=_request(),
        selections=selections,
        graph=_graph(),
        sdmx=sdmx,
        rate_cache=RateTableCache(),
    )
    assert plan is not None
    assert len(server.data_requests()) == 2  # exchange-rate and deflator, each one fetch

    rows = _member_rows([_row(provider="country/FRA", year="2020", value="500")])
    table, record = apply_member_conversion(plan, rows, declared_roles=_ROLES)

    expected = 500.0 * (100.0 / 110.0) * 1.0 / 1.0
    assert table["OBS_VALUE"].to_pylist() == [repr(expected)]
    assert record is not None
    assert record.deflator_table_version is not None


def test_second_plan_for_the_same_recipe_and_version_hits_the_cache():
    unit_request = UnitRequest(currency="EUR", prices="current")
    selections = _selections(native_unit="CurrencyUSD", request=unit_request)
    sdmx, server = _multi_sdmx(
        {
            _EXCHANGE_VAR: RATE_CSV_HEADER
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2022", value="1.0")
            + _rate_row(
                stat_var=_EXCHANGE_VAR,
                entity="CurrencyEUR",
                year="2022",
                value="0.9",
            )
        }
    )
    cache = RateTableCache()
    graph = _graph()

    plan_member_conversion(
        request=_request(), selections=selections, graph=graph, sdmx=sdmx, rate_cache=cache
    )
    assert len(server.data_requests()) == 1

    plan_member_conversion(
        request=_request(), selections=selections, graph=graph, sdmx=sdmx, rate_cache=cache
    )
    assert len(server.data_requests()) == 1  # the second plan's load_table call hits the cache


def test_a_uniform_plan_loads_every_entity_the_tables_serve_whatever_the_bindings():
    unit_request = UnitRequest(currency="USD", prices="constant", base_year=2020)
    selections = _selections(native_unit="CurrencyUSD", request=unit_request)
    sdmx, server = _multi_sdmx(
        {
            _EXCHANGE_VAR: RATE_CSV_HEADER
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2020", value="1.0")
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyKES", year="2020", value="112.0"),
            _DEFLATOR_VAR: RATE_CSV_HEADER
            + _rate_row(stat_var=_DEFLATOR_VAR, entity="country/FRA", year="2020", value="100.0")
            + _rate_row(stat_var=_DEFLATOR_VAR, entity="country/DEU", year="2020", value="101.0")
            + _rate_row(
                stat_var=_DEFLATOR_VAR, entity="DACCountriesTotal", year="2020", value="100.5"
            ),
        }
    )

    plan = plan_member_conversion(
        request=_request(role_bindings={"financingProvider": ["country/FRA"]}),
        selections=selections,
        graph=_graph(),
        sdmx=sdmx,
        rate_cache=RateTableCache(),
    )

    assert plan is not None
    assert plan.exchange_rate_table is not None
    assert plan.deflator_table is not None
    assert len(server.availability_requests()) == 2
    observed = [query_params(request)["c[observationAbout]"] for request in server.data_requests()]
    assert observed == ["CurrencyUSD,CurrencyKES", "country/FRA,country/DEU,DACCountriesTotal"]
    assert plan.deflator_table.table["entity"].to_pylist() == [
        "DACCountriesTotal",
        "country/DEU",
        "country/FRA",
    ]


def test_a_uniform_and_a_role_shaped_plan_share_one_cached_table():
    sdmx, server = _multi_sdmx(
        {
            _EXCHANGE_VAR: RATE_CSV_HEADER
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2022", value="1.0")
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyKES", year="2022", value="112.0")
        }
    )
    cache = RateTableCache()
    graph = _graph(arcs=_with_entity_currency("country/KEN", "CurrencyKES"))

    uniform = plan_member_conversion(
        request=_request(),
        selections=_selections(
            native_unit="CurrencyUSD", request=UnitRequest(currency="KES", prices="current")
        ),
        graph=graph,
        sdmx=sdmx,
        rate_cache=cache,
    )
    role_shaped = plan_member_conversion(
        request=_request(),
        selections=_selections(
            native_unit="CurrencyUSD",
            request=UnitRequest(currency="destination", prices="current"),
        ),
        graph=graph,
        sdmx=sdmx,
        rate_cache=cache,
    )
    assert uniform is not None
    assert role_shaped is not None
    rows = _member_rows([_row(recipient="country/KEN", year="2022", value="100")])
    uniform_table, _ = apply_member_conversion(uniform, rows, declared_roles=_ROLES)
    role_table, _ = _convert(role_shaped, rows)

    assert len(server.data_requests()) == 1
    assert uniform_table["OBS_VALUE"].to_pylist() == role_table["OBS_VALUE"].to_pylist()


def test_current_price_conversion_needs_no_deflator_role_match():
    # Current-price conversion needs no deflator role. `financingRecipient` is outside the DAC set.
    unit_request = UnitRequest(currency="EUR", prices="current")
    selections = _selections(native_unit="CurrencyUSD", request=unit_request)
    sdmx, server = _multi_sdmx(
        {
            _EXCHANGE_VAR: RATE_CSV_HEADER
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2022", value="1.0")
            + _rate_row(
                stat_var=_EXCHANGE_VAR,
                entity="CurrencyEUR",
                year="2022",
                value="0.9",
            )
        }
    )

    plan = plan_member_conversion(
        request=_request(declared_roles=("financingRecipient",)),
        selections=selections,
        graph=_graph(),
        sdmx=sdmx,
        rate_cache=RateTableCache(),
    )
    assert plan is not None
    assert plan.recipe is not None
    assert len(server.data_requests()) == 1


def test_constant_price_conversion_refuses_when_no_deflator_role_matches():
    # Constant prices require a declared deflator role.
    unit_request = UnitRequest(currency="EUR", prices="constant", base_year=2015)
    selections = _selections(native_unit="CurrencyUSD", request=unit_request)
    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        plan_member_conversion(
            request=_request(declared_roles=("financingRecipient",)),
            selections=selections,
            graph=_graph(),
            sdmx=_multi_sdmx({})[0],
            rate_cache=RateTableCache(),
        )
    assert exc_info.value.cause == "role_undeclared"


def test_eur_native_converts_to_gbp_current_without_refusal():
    # The loader adds the USD anchor for non-USD native and target currencies.
    unit_request = UnitRequest(currency="GBP", prices="current")
    selections = _selections(native_unit="CurrencyEUR", request=unit_request)
    sdmx, server = _multi_sdmx(
        {
            _EXCHANGE_VAR: RATE_CSV_HEADER
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2022", value="1.0")
            + _rate_row(
                stat_var=_EXCHANGE_VAR,
                entity="CurrencyEUR",
                year="2022",
                value="0.9",
            )
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyGBP", year="2022", value="0.8")
        }
    )

    plan = plan_member_conversion(
        request=_request(),
        selections=selections,
        graph=_graph(),
        sdmx=sdmx,
        rate_cache=RateTableCache(),
    )
    assert plan is not None
    assert len(server.data_requests()) == 1

    rows = _member_rows([_row(year="2022", value="100")])
    table, record = apply_member_conversion(plan, rows, declared_roles=_ROLES)
    assert record is not None
    assert table["OBS_VALUE"].to_pylist() == [repr(100.0 * 0.8 / 0.9)]


def test_role_shaped_destination_currency_resolves_served_entities_only():
    # Role-shaped currency uses a spec role key such as `destination`.
    unit_request = UnitRequest(currency="destination", prices="current")
    selections = _selections(native_unit="CurrencyUSD", request=unit_request)
    sdmx, server = _multi_sdmx(
        {
            _EXCHANGE_VAR: RATE_CSV_HEADER
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2022", value="1.0")
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyKES", year="2022", value="112.0")
        }
    )
    arcs = _with_entity_currency("country/KEN", "CurrencyKES")

    plan = plan_member_conversion(
        request=_request(
            role_bindings={
                "financingProvider": ["country/FRA"],
                # The bound organisation has no row in this response.
                "financingRecipient": ["country/KEN", "OrgWithNoCurrentCurrency"],
            }
        ),
        selections=selections,
        graph=_graph(arcs=arcs),
        sdmx=sdmx,
        rate_cache=RateTableCache(),
    )
    assert plan is not None
    assert plan.spec_role == "destination"
    assert server.requests == []  # role-shaped tables load lazily, inside apply

    rows = _member_rows([_row(recipient="country/KEN", year="2022", value="100")])
    table, record = _convert(plan, rows)

    assert len(server.data_requests()) == 1
    assert table["OBS_VALUE"].to_pylist() == [repr(100.0 * 112.0 / 1.0)]
    assert table["unit"].to_pylist() == [
        {"currency": "KES", "prices": "current", "base_year": None}
    ]
    assert record is not None


def test_role_shaped_constant_prices_uses_the_recipe_fallback_for_a_missing_deflator_row():
    unit_request = UnitRequest(currency="destination", prices="constant", base_year=2020)
    selections = _selections(native_unit="CurrencyUSD", request=unit_request)
    arcs = _with_entity_currency("country/KEN", "CurrencyKES")
    sdmx, server = _multi_sdmx(
        {
            _EXCHANGE_VAR: RATE_CSV_HEADER
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2020", value="1.0")
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyKES", year="2020", value="112.0"),
            _DEFLATOR_VAR: RATE_CSV_HEADER
            + _rate_row(stat_var=_DEFLATOR_VAR, entity="country/FRA", year="2020", value="100.0")
            + _rate_row(
                stat_var=_DEFLATOR_VAR, entity="DACCountriesTotal", year="2020", value="105.0"
            ),
        }
    )

    plan = plan_member_conversion(
        request=_request(
            role_bindings={
                "financingProvider": ["country/FRA", "country/ZAF"],
                "financingRecipient": ["country/KEN"],
            }
        ),
        selections=selections,
        graph=_graph(arcs=arcs),
        sdmx=sdmx,
        rate_cache=RateTableCache(),
    )
    assert plan is not None
    assert server.requests == []  # role-shaped tables load lazily, inside apply

    # The member row for ZAF has no deflator; conversion uses the recipe fallback.
    rows = _member_rows(
        [
            _row(provider="country/FRA", recipient="country/KEN", year="2020", value="100"),
            _row(provider="country/ZAF", recipient="country/KEN", year="2020", value="50"),
        ]
    )
    table, record = _convert(plan, rows)

    assert len(server.data_requests()) == 2  # exchange-rate and deflator, each one fetch
    expected_fra = 100.0 * (100.0 / 100.0) * 112.0 / 1.0
    expected_zaf = 50.0 * (105.0 / 105.0) * 112.0 / 1.0
    assert table["OBS_VALUE"].to_pylist() == [repr(expected_fra), repr(expected_zaf)]
    assert record is not None
    assert record.fallback_count == 1
    assert record.fallback_examples == (("country/ZAF", "DACCountriesTotal"),)


def test_second_role_shaped_constant_price_plan_hits_both_table_caches():
    unit_request = UnitRequest(currency="destination", prices="constant", base_year=2020)
    selections = _selections(native_unit="CurrencyUSD", request=unit_request)
    arcs = _with_entity_currency("country/KEN", "CurrencyKES")
    sdmx, server = _multi_sdmx(
        {
            _EXCHANGE_VAR: RATE_CSV_HEADER
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2020", value="1.0")
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyKES", year="2020", value="112.0"),
            _DEFLATOR_VAR: RATE_CSV_HEADER
            + _rate_row(stat_var=_DEFLATOR_VAR, entity="country/FRA", year="2020", value="100.0"),
        }
    )
    cache = RateTableCache()
    graph = _graph(arcs=arcs)
    request = _request(
        role_bindings={
            "financingProvider": ["country/FRA"],
            "financingRecipient": ["country/KEN"],
        }
    )
    rows = _member_rows(
        [_row(provider="country/FRA", recipient="country/KEN", year="2020", value="100")]
    )

    plan_one = plan_member_conversion(
        request=request, selections=selections, graph=graph, sdmx=sdmx, rate_cache=cache
    )
    _convert(plan_one, rows)
    assert len(server.data_requests()) == 2  # one exchange-rate fetch, one deflator fetch

    plan_two = plan_member_conversion(
        request=request, selections=selections, graph=graph, sdmx=sdmx, rate_cache=cache
    )
    _convert(plan_two, rows)
    assert len(server.data_requests()) == 2  # both tables reused from the cache, no new fetch


def test_role_shaped_target_keeps_struct_type_on_a_zero_row_table():
    unit_request = UnitRequest(currency="origin", prices="current")
    selections = _selections(native_unit="CurrencyUSD", request=unit_request)
    arcs = _with_entity_currency("country/FRA", "CurrencyEUR")
    sdmx, _ = _multi_sdmx(
        {
            _EXCHANGE_VAR: RATE_CSV_HEADER
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2022", value="1.0")
        }
    )

    plan = plan_member_conversion(
        request=_request(),
        selections=selections,
        graph=_graph(arcs=arcs),
        sdmx=sdmx,
        rate_cache=RateTableCache(),
    )
    assert plan is not None

    empty_rows = _member_rows([])
    table, record = _convert(plan, empty_rows)

    assert table.num_rows == 0
    assert pa.types.is_struct(table.schema.field("unit").type)
    assert record is not None  # convert_member_marked always returns a record, even over zero rows


def test_role_shaped_eur_native_converts_to_a_destination_currency_without_refusal():
    # The exchange-rate load includes the USD anchor for EUR-to-KES conversion.
    unit_request = UnitRequest(currency="destination", prices="current")
    selections = _selections(native_unit="CurrencyEUR", request=unit_request)
    arcs = _with_entity_currency("country/KEN", "CurrencyKES")
    sdmx, server = _multi_sdmx(
        {
            _EXCHANGE_VAR: RATE_CSV_HEADER
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2022", value="1.0")
            + _rate_row(
                stat_var=_EXCHANGE_VAR,
                entity="CurrencyEUR",
                year="2022",
                value="0.9",
            )
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyKES", year="2022", value="112.0")
        }
    )

    plan = plan_member_conversion(
        request=_request(),
        selections=selections,
        graph=_graph(arcs=arcs),
        sdmx=sdmx,
        rate_cache=RateTableCache(),
    )
    assert plan is not None
    assert server.requests == []  # role-shaped tables load lazily, inside apply

    rows = _member_rows([_row(recipient="country/KEN", year="2022", value="100")])
    table, record = _convert(plan, rows)

    assert len(server.data_requests()) == 1
    assert record is not None
    assert table["OBS_VALUE"].to_pylist() == [repr(100.0 * 112.0 / 0.9)]


def test_withdrawn_native_currency_refuses_rate_missing_at_conversion():
    """A withdrawn currency is still a `CurrencyUnit`. The rate table has no
    row for it, so conversion refuses naming the currency-year."""
    unit_request = UnitRequest(currency="EUR", prices="current")
    selections = _selections(native_unit="CurrencyFRF", request=unit_request)
    sdmx, _ = _multi_sdmx(
        {
            _EXCHANGE_VAR: RATE_CSV_HEADER
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2022", value="1.0")
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyEUR", year="2022", value="0.9")
        }
    )
    plan = plan_member_conversion(
        request=_request(),
        selections=selections,
        graph=_graph(),
        sdmx=sdmx,
        rate_cache=RateTableCache(),
    )
    with pytest.raises(RateCoverageGapError) as exc_info:
        apply_member_conversion(
            plan, _member_rows([_row(year="2022", value="100")]), declared_roles=_ROLES
        )
    assert exc_info.value.cause == "rate_missing"
    assert exc_info.value.entities == ("CurrencyFRF@2022",)


@pytest.mark.parametrize(
    ("recipients", "served", "omitted"),
    [
        pytest.param(
            ["country/KEN", "southOfSahara", "africaRegion"],
            ["country/KEN"],
            ("africaRegion", "southOfSahara"),
            id="some_unmapped",
        ),
        pytest.param(
            ["southOfSahara", "africaRegion"],
            [],
            ("africaRegion", "southOfSahara"),
            id="all_unmapped",
        ),
    ],
)
def test_role_shaped_target_omits_the_rows_of_unmapped_served_entities(
    recipients: list[str], served: list[str], omitted: tuple[str, ...]
):
    """Regional recipients have no national currency: their rows are dropped before
    conversion and the record names them in dcid order. Kenya's row converts. Whether a
    request left with no row refuses is the read's decision, not this one's."""
    unit_request = UnitRequest(currency="destination", prices="current")
    sdmx, _server = _multi_sdmx(
        {
            _EXCHANGE_VAR: RATE_CSV_HEADER
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2022", value="1.0")
            + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyKES", year="2022", value="112.0")
        }
    )
    plan = plan_member_conversion(
        request=_request(
            role_bindings={"financingProvider": ["country/FRA"], "financingRecipient": recipients}
        ),
        selections=_selections(native_unit="CurrencyUSD", request=unit_request),
        graph=_graph(arcs=_with_entity_currency("country/KEN", "CurrencyKES")),
        sdmx=sdmx,
        rate_cache=RateTableCache(),
    )
    rows = _member_rows([_row(recipient=r, year="2022", value="100") for r in recipients])

    table, record = _convert(plan, rows)

    assert table["financingRecipient"].to_pylist() == served
    assert table["OBS_VALUE"].to_pylist() == [repr(100.0 * 112.0)] * len(served)
    assert pa.types.is_struct(table.schema.field("unit").type)
    assert record is not None
    assert record.omitted_entities == omitted


def test_missing_nominal_qualifier_refuses_recipe_not_declared():
    unit_request = UnitRequest(currency="EUR", prices="current")
    selections = _selections(native_unit="CurrencyUSD", request=unit_request)
    arcs = _base_arcs(measurementQualifier=())
    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        plan_member_conversion(
            request=_request(),
            selections=selections,
            graph=_graph(arcs),
            sdmx=_multi_sdmx({})[0],
            rate_cache=RateTableCache(),
        )
    assert exc_info.value.cause == "nominal_undeclared"


def test_missing_recipe_refuses_recipe_not_declared():
    unit_request = UnitRequest(currency="EUR", prices="current")
    selections = _selections(
        native_unit="CurrencyUSD", provenance="provenance/Orphan", request=unit_request
    )
    arcs = _base_arcs()
    arcs["provenance/Orphan"] = {"conversionRecipe": (), "isPartOf": ()}
    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        plan_member_conversion(
            request=_request(),
            selections=selections,
            graph=_graph(arcs),
            sdmx=_multi_sdmx({})[0],
            rate_cache=RateTableCache(),
        )
    assert exc_info.value.cause == "recipe_undeclared"


@pytest.mark.parametrize(
    ("native_unit", "target"),
    [
        pytest.param("CurrencyXXX", "EUR", id="native-typed-otherwise"),
        pytest.param("CurrencyUSD", "XXX", id="target-not-held"),
    ],
)
def test_non_currency_native_or_target_unit_refuses(native_unit: str, target: str):
    unit_request = UnitRequest(currency=target, prices="current")
    selections = _selections(native_unit=native_unit, request=unit_request)
    arcs = _base_arcs()
    arcs["CurrencyXXX"] = {"typeOf": ("StatisticalVariable",)}
    if target == "XXX":
        del arcs["CurrencyXXX"]
    with pytest.raises(NonCurrencyUnitError):
        plan_member_conversion(
            request=_request(),
            selections=selections,
            graph=_graph(arcs),
            sdmx=_multi_sdmx({})[0],
            rate_cache=RateTableCache(),
        )


def test_role_shaped_undeclared_axis_refuses_at_plan_time_before_any_fetch():
    # The shape declares only `observationAbout`; `origin` is unsupported.
    unit_request = UnitRequest(currency="origin", prices="current")
    selections = _selections(native_unit="CurrencyUSD", request=unit_request)
    sdmx, server = _multi_sdmx({})
    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        plan_member_conversion(
            request=_request(declared_roles=("observationAbout",)),
            selections=selections,
            graph=_graph(),
            sdmx=sdmx,
            rate_cache=RateTableCache(),
        )
    assert exc_info.value.cause == "role_undeclared"
    assert server.requests == []


def test_role_shaped_constant_price_undeclared_deflator_role_refuses_on_both_decision_paths():
    # `financingRecipient` is declared, so axis resolution succeeds. The DAC recipe requires
    # `financingProvider` or `observationAbout` for deflation, so both paths refuse before fetching.
    unit_request = UnitRequest(currency="destination", prices="constant", base_year=2020)
    graph = _graph()

    selections = _selections(native_unit="CurrencyUSD", request=unit_request)
    plan_sdmx, plan_server = _multi_sdmx({})
    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        plan_member_conversion(
            request=_request(declared_roles=("financingRecipient",)),
            selections=selections,
            graph=graph,
            sdmx=plan_sdmx,
            rate_cache=RateTableCache(),
        )
    assert exc_info.value.cause == "role_undeclared"
    assert plan_server.requests == []

    facet = Facet(
        facet_id="f1",
        provenance_id=_MEMBER_PROVENANCE,
        measurement_method="NotApplicable",
        unit="CurrencyUSD",
        obs_count=5,
        earliest_date="2020",
        latest_date="2020",
    )
    entry_request = MaterialisationRequest(
        stat_var=_STAT_VAR,
        role_bindings={
            "financingProvider": ["country/FRA"],
            "financingRecipient": ["country/KEN"],
        },
        declared_roles=("financingRecipient",),
        facet_bindings={"unit": FacetBinding(state="request", request=unit_request)},
        window=RequestedWindow(kind="all"),
    )
    entry_sdmx, entry_server = _multi_sdmx({})
    with pytest.raises(RecipeNotDeclaredError) as exc_info2:
        decide_member_unit(
            entry_request,
            settle=True,
            graph=graph,
            sdmx=entry_sdmx,
            coverage=_UNUSED_COVERAGE,
            precomputed=[facet],
        )
    assert exc_info2.value.cause == "role_undeclared"
    assert (
        entry_server.requests == []
    )  # Precomputed facets skip probing, so the decision uses no SDMX data.


def test_role_shaped_constant_price_undeclared_deflator_role_wins_over_an_unmapped_served_currency():
    # Deflator-role validation precedes currency mapping during conversion.
    unit_request = UnitRequest(currency="destination", prices="constant", base_year=2020)
    selections = _selections(native_unit="CurrencyUSD", request=unit_request)
    sdmx, server = _multi_sdmx({})

    with pytest.raises(RecipeNotDeclaredError) as exc_info:
        plan_member_conversion(
            request=_request(
                declared_roles=("financingRecipient",),
                role_bindings={
                    "financingProvider": ["country/FRA"],
                    "financingRecipient": ["OrgWithNoCurrentCurrency"],
                },
            ),
            selections=selections,
            graph=_graph(),
            sdmx=sdmx,
            rate_cache=RateTableCache(),
        )
    assert exc_info.value.cause == "role_undeclared"
    assert server.requests == []


def test_apply_refuses_a_table_whose_unit_column_is_already_a_struct():
    rows = _member_rows([_row(year="2022", value="100")])
    struct_type = pa.struct(
        [("currency", pa.string()), ("prices", pa.string()), ("base_year", pa.int64())]
    )
    struct_unit = pa.array(
        [{"currency": "USD", "prices": "current", "base_year": None}], type=struct_type
    )
    rows = rows.set_column(rows.schema.get_field_index("unit"), "unit", struct_unit)
    with pytest.raises(MaterialisationUsageError):
        apply_member_conversion(None, rows, declared_roles=_ROLES)


def test_apply_refuses_when_declared_roles_disagrees_with_the_plans_own():
    unit_request = UnitRequest(currency="USD", prices="current")
    selections = _selections(native_unit="CurrencyUSD", request=unit_request)
    plan = plan_member_conversion(
        request=_request(),
        selections=selections,
        graph=_graph(),
        sdmx=_multi_sdmx({})[0],
        rate_cache=RateTableCache(),
    )
    assert plan is not None

    rows = _member_rows([_row(year="2022", value="100")])
    with pytest.raises(MaterialisationUsageError):
        apply_member_conversion(plan, rows, declared_roles=("observationAbout",))


@pytest.mark.parametrize(
    ("unit_request", "years"),
    [
        (UnitRequest(currency="EUR", prices="current"), ("2020", "2021")),
        (UnitRequest(currency="EUR", prices="constant", base_year=1950), ("2020",)),
        (UnitRequest(currency="destination", prices="constant", base_year=2040), ("2020",)),
    ],
)
def test_the_coverage_check_refuses_as_the_conversion_of_the_same_rows(unit_request, years):
    """`require_decision_coverage` over the rows' entities and years, under the member's
    `coverage_decision`, raises the refusal `apply_member_conversion` raises converting those
    rows: the same lookup, before a read."""
    tables = {
        _EXCHANGE_VAR: RATE_CSV_HEADER
        + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyUSD", year="2020", value="1.0")
        + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyEUR", year="2020", value="0.9")
        + _rate_row(stat_var=_EXCHANGE_VAR, entity="CurrencyKES", year="2020", value="112.0"),
        _DEFLATOR_VAR: RATE_CSV_HEADER
        + _rate_row(stat_var=_DEFLATOR_VAR, entity="country/FRA", year="2020", value="100.0"),
    }
    selections = _selections(native_unit="CurrencyUSD", request=unit_request)
    arcs = _with_entity_currency("country/KEN", "CurrencyKES")
    rows = [_row(year=year, value="100") for year in years]

    sdmx, _server = _multi_sdmx(tables)
    plan = plan_member_conversion(
        request=_request(),
        selections=selections,
        graph=_graph(arcs=arcs),
        sdmx=sdmx,
        rate_cache=RateTableCache(),
    )
    with pytest.raises(RateCoverageGapError) as converting:
        _convert(plan, _member_rows(rows))

    sdmx, server = _multi_sdmx(tables)
    decision = coverage_decision(
        stat_var=_STAT_VAR, declared_roles=_ROLES, selections=selections, graph=_graph(arcs=arcs)
    )
    assert decision is not None
    with pytest.raises(RateCoverageGapError) as checking:
        require_decision_coverage(
            decision,
            served=[
                served_block(
                    int(row["TIME_PERIOD"]),
                    {
                        "financingProvider": [row["financingProvider"]],
                        "financingRecipient": [row["financingRecipient"]],
                    },
                )
                for row in rows
            ],
            graph=_graph(arcs=arcs),
            sdmx=sdmx,
            rate_cache=RateTableCache(),
        )
    assert (checking.value.cause, checking.value.entities) == (
        converting.value.cause,
        converting.value.entities,
    )
    assert {query_params(r)["c[variableMeasured]"] for r in server.requests} <= set(tables)


_SERIES_YEARS = range(2010, 2023)
_FALLBACK_PROVIDERS = tuple(f"country/P{index:02d}" for index in range(12))


def _many_series_plan(unit_request: UnitRequest) -> MemberConversionPlan:
    """A DAC plan whose tables project France's deflator in every year, and the DAC total's
    deflator and the euro and shilling rates in 2022, and hold no deflator for
    `_FALLBACK_PROVIDERS`."""
    rates = {"CurrencyUSD": 1.0, "CurrencyEUR": 0.9, "CurrencyKES": 112.0, "CurrencyUGX": 3700.0}
    sdmx, _server = _multi_sdmx(
        recipe_table_bodies(
            _DAC_RECIPE,
            rates={dcid: dict.fromkeys(_SERIES_YEARS, rate) for dcid, rate in rates.items()},
            deflators={
                dcid: dict.fromkeys(_SERIES_YEARS, 100.0)
                for dcid in ("country/FRA", "country/DEU", "DACCountriesTotal")
            },
            projected=[
                *(("country/FRA", year) for year in _SERIES_YEARS),
                ("DACCountriesTotal", 2022),
                ("CurrencyEUR", 2022),
                ("CurrencyKES", 2022),
            ],
        )
    )
    arcs = _with_entity_currency("country/KEN", "CurrencyKES")
    arcs["CurrencyUGX"] = {"typeOf": ("CurrencyUnit",), "nationalCurrencyOf": ("country/UGA",)}
    plan = plan_member_conversion(
        request=_request(),
        selections=_selections(native_unit="CurrencyUSD", request=unit_request),
        graph=_graph(arcs=arcs),
        sdmx=sdmx,
        rate_cache=RateTableCache(),
    )
    assert plan is not None
    return plan


def _many_series_rows() -> pa.Table:
    """France to Kenya in every year under facet f2, with one contested 2010 row under f1;
    Germany to Uganda once; France to the region `southOfSahara`, which no single currency
    names, once; each fallback provider to Kenya in 2016 and 2017, the first also in 2022.
    Facet f1 holds more of the member's rows and f2 more of France's series."""
    rows = [{**_row(year=str(year), value="100"), "facetId": "f2"} for year in _SERIES_YEARS]
    rows.append({**_row(year="2010", value="999"), "facetId": "f1"})
    rows.append({**_row(provider="country/DEU", recipient="country/UGA", year="2016", value="7")})
    rows.append(_row(recipient="southOfSahara", year="2016", value="5"))
    rows += [
        _row(provider=provider, year=year, value="10")
        for provider in _FALLBACK_PROVIDERS
        for year in ("2016", "2017")
    ]
    rows.append(_row(provider=_FALLBACK_PROVIDERS[0], year="2022", value="10"))
    return _member_rows([{"facetId": "f1", **row} for row in rows])


def _by_series_and_period(table: pa.Table) -> pa.Table:
    return table.sort_by([(column, "ascending") for column in (*_ROLES, "TIME_PERIOD")])


@pytest.mark.parametrize(
    "unit_request",
    [
        UnitRequest(currency="EUR", prices="constant", base_year=2015),
        UnitRequest(currency="destination", prices="constant", base_year=2015),
        UnitRequest(currency="EUR", prices="current"),
        UnitRequest(currency="destination", prices="current"),
    ],
    ids=repr,
)
def test_a_mint_converts_a_member_as_each_series_converts_alone(unit_request: UnitRequest):
    """`_convert_by_series` returns the rows, member record and per-series records that
    converting each series on its own gives, `apply_member_conversion` over each
    `_split_by_roles` part: facets rank within a series, and fallback and projected markers,
    their counts, sorted capped examples and zeros stay with their own series. A
    role-shaped target names the region it omitted on the member's and every series'
    record."""
    plan = _many_series_plan(unit_request)
    rows, currencies = omit_unmapped_rows(plan, _many_series_rows(), served=_many_series_rows())

    converted, record, series = _convert_by_series(
        plan, rows, request=_request(), currencies=currencies
    )

    parts: list[pa.Table] = []
    records = {}
    for dcids, part in _split_by_roles(rows, _ROLES):
        part_converted, part_record = apply_member_conversion(
            plan, part, declared_roles=_ROLES, currencies=currencies
        )
        parts.append(part_converted)
        records[dcids] = part_record
    assert _by_series_and_period(converted).equals(_by_series_and_period(pa.concat_tables(parts)))
    assert record == merged_record([r for r in records.values() if r is not None])
    assert series == {
        series_key(_STAT_VAR, _ROLES, dcids): SeriesConversion.from_record(r)
        for dcids, r in records.items()
        if r is not None
    }
    france = series[series_key(_STAT_VAR, _ROLES, ("country/FRA", "country/KEN"))]
    germany = series[series_key(_STAT_VAR, _ROLES, ("country/DEU", "country/UGA"))]
    omitted = ("southOfSahara",) if unit_request.currency == "destination" else ()
    assert record is not None
    assert record.omitted_entities == omitted
    _rows, _record, series_records = apply_member_conversion_by_series(
        plan, rows, declared_roles=_ROLES, currencies=currencies
    )
    assert series_records == records
    assert (germany.fallback_count, germany.projected_count) == (0, 0)
    assert france.projected_count == (1 if unit_request.prices == "current" else 13)
    if unit_request.prices == "constant":
        assert len(france.projected_examples) == _MAX_NAMED_ENTITIES
        assert record.fallback_count == 25
        assert len(record.fallback_examples) == _MAX_NAMED_ENTITIES


def test_a_mint_converts_a_many_series_member_in_one_call(monkeypatch: pytest.MonkeyPatch):
    """A member of 200 single-row series is collapsed and converted in one pass over its
    rows, where splitting it applies the conversion once per series."""
    calls: list[int] = []
    conversions: list[int] = []
    collapse = member_conversion.collapse_duplicate_identity_facets_table
    convert = member_conversion.convert_member_marked

    def counting_collapse(table: pa.Table, **kwargs):
        calls.append(table.num_rows)
        return collapse(table, **kwargs)

    def counting_convert(*, rows: pa.Table, **kwargs):
        conversions.append(rows.num_rows)
        return convert(rows=rows, **kwargs)

    monkeypatch.setattr(
        member_conversion, "collapse_duplicate_identity_facets_table", counting_collapse
    )
    monkeypatch.setattr(member_conversion, "convert_member_marked", counting_convert)
    plan = _many_series_plan(UnitRequest(currency="EUR", prices="current"))
    providers = [f"country/Q{index:03d}" for index in range(200)]
    rows = _member_rows([_row(provider=provider, year="2020", value="1") for provider in providers])

    _converted, _record, series = _convert_by_series(
        plan, rows, request=_request(), currencies=None
    )

    assert calls == [200]
    assert conversions == [200]
    assert len(series) == 200
