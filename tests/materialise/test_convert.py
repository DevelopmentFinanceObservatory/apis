"""Conversion formula, coverage gaps, projection rules, and scalar/vector parity."""

from __future__ import annotations

import dataclasses
import random
import struct
from collections.abc import Mapping, Sequence
from typing import cast

import pyarrow as pa
import pytest

from dc_sdmx.results import VALUE_COLUMN
from materialise.columnar import observed_values
from materialise.convert import (
    ConversionRecord,
    ResolvedUnitRequest,
    RoleShapedTarget,
    UniformTarget,
    convert_member_marked,
    convert_scalar,
    merged_record,
)
from materialise.errors import MaterialisationUsageError, UpstreamResponseError
from materialise.rate_tables import RATE_TABLE_SCHEMA, RateTable, RateTableKind
from materialise.recipe_bodies import RECIPE_BODIES
from materialise.recipes import ConversionRecipe, RateCoverageGapError

from ._table_support import table_from_cells

_DAC = RECIPE_BODIES["ConversionRecipeDAC"]
_IMF = RECIPE_BODIES["ConversionRecipeIMF"]
_IDS = RECIPE_BODIES["ConversionRecipeIDS"]
_FLOW_STAT_VAR = "CRS_Nominal_Flow"
_ROLES = ("financingProvider", "financingRecipient")

_USD = "CurrencyUSD"
_EUR = "CurrencyEUR"
_KES = "CurrencyKES"
_GBP = "CurrencyGBP"


def _recipe(
    base: ConversionRecipe = _DAC,
    *,
    fallback_entity: str | None = None,
    forecast_policy: str = "FlagProjected",
) -> ConversionRecipe:
    return dataclasses.replace(
        base, fallback_entity=fallback_entity, forecast_policy=forecast_policy
    )


def _rate_table(
    *,
    kind: RateTableKind,
    rows: Sequence[tuple[str, int, float, str]],
    version: str = "digest",
) -> RateTable:
    table = pa.Table.from_pylist(
        [{"entity": e, "year": y, "value": v, "facet": f} for e, y, v, f in rows],
        schema=RATE_TABLE_SCHEMA,
    )
    stat_var = _DAC.exchange_rate_variable if kind == "exchange_rate" else _DAC.deflator_variable
    return RateTable(
        kind=kind,
        stat_var=stat_var,
        provenance_id=_DAC.table_provenance,
        version=f"{kind}-{version}",
        refresh_date="2026-01-01",
        table=table,
    )


def _rates(*rows: tuple[str, int, float, str]) -> RateTable:
    return _rate_table(kind="exchange_rate", rows=rows)


def _deflators(*rows: tuple[str, int, float, str]) -> RateTable:
    return _rate_table(kind="deflator", rows=rows)


def _member_rows(
    records: Sequence[Mapping[str, str | None]], *, roles: Sequence[str] = _ROLES
) -> pa.Table:
    return table_from_cells(records, roles=roles)


def _row(
    *,
    provider: str = "country/FRA",
    recipient: str = "country/KEN",
    year: str,
    value: str,
) -> dict[str, str | None]:
    return {
        "variableMeasured": _FLOW_STAT_VAR,
        "financingProvider": provider,
        "financingRecipient": recipient,
        "TIME_PERIOD": year,
        "OBS_VALUE": value,
    }


def _convert(
    rows: pa.Table,
    *,
    target: UniformTarget | RoleShapedTarget,
    exchange: RateTable,
    native: str = _USD,
    recipe: ConversionRecipe | None = None,
    deflator: RateTable | None = None,
    base_year: int | None = None,
    declared_roles: tuple[str, ...] = _ROLES,
):
    prices = "current" if base_year is None and deflator is None else "constant"
    converted, record, _markers = convert_member_marked(
        rows=rows,
        unit_request=ResolvedUnitRequest(prices=prices, target=target, base_year=base_year),
        recipe=recipe if recipe is not None else _recipe(),
        native_currency=native,
        native_unit_ref=native,
        declared_roles=declared_roles,
        exchange_rate_table=exchange,
        deflator_table=deflator,
    )
    return converted, record


def _bit_exact(a: float, b: float) -> bool:
    return struct.pack("<d", a) == struct.pack("<d", b)


# ---------------------------------------------------------------------------
# The formula against a hand computation
# ---------------------------------------------------------------------------


def test_current_prices_formula_matches_hand_computation() -> None:
    rows = _member_rows([_row(year="2021", value="400")])
    exchange = _rates((_USD, 2021, 1.0, "actual"), (_KES, 2021, 112.0, "actual"))
    converted, record = _convert(rows, target=UniformTarget(_KES), exchange=exchange)
    assert converted[VALUE_COLUMN].to_pylist()[0] == repr(400.0 * 112.0 / 1.0)
    assert converted["unit"].to_pylist()[0] == {
        "currency": "KES",
        "prices": "current",
        "base_year": None,
    }
    assert record.deflator_table_version is None
    assert record.native_currency == _USD
    assert record.table_provenance_id == "provenance/ONEConversionTablesDAC"
    assert record.fallback_count == 0
    assert record.projected_count == 0


def test_constant_prices_formula_matches_hand_computation() -> None:
    """`V x (D[R,b] / D[R,t]) x X[T,b] / X[S,t]`, R the provider."""
    rows = _member_rows([_row(year="2020", value="1000")])
    exchange = _rates((_EUR, 2020, 0.85, "actual"), (_GBP, 2015, 0.65, "actual"))
    deflator = _deflators(
        ("country/FRA", 2015, 100.0, "actual"), ("country/FRA", 2020, 120.0, "actual")
    )
    converted, record = _convert(
        rows,
        target=UniformTarget(_GBP),
        exchange=exchange,
        native=_EUR,
        deflator=deflator,
        base_year=2015,
    )
    assert converted[VALUE_COLUMN].to_pylist()[0] == repr(1000.0 * (100.0 / 120.0) * 0.65 / 0.85)
    assert converted["unit"].to_pylist()[0] == {
        "currency": "GBP",
        "prices": "constant",
        "base_year": 2015,
    }
    assert record.deflator_table_version == deflator.version
    assert record.exchange_rate_table_version == exchange.version


def test_the_deflator_entity_needs_no_exchange_rate_row() -> None:
    """R appears only in the deflator terms, so a provider whose currency the
    rate table lacks still converts."""
    rows = _member_rows([_row(provider="country/BEN", year="2020", value="50")])
    exchange = _rates((_USD, 2020, 1.0, "actual"), (_KES, 2015, 140.0, "actual"))
    deflator = _deflators(
        ("country/BEN", 2015, 90.0, "actual"), ("country/BEN", 2020, 110.0, "actual")
    )
    converted, _ = _convert(
        rows, target=UniformTarget(_KES), exchange=exchange, deflator=deflator, base_year=2015
    )
    assert converted[VALUE_COLUMN].to_pylist()[0] == repr(50.0 * (90.0 / 110.0) * 140.0 / 1.0)


def test_ids_recipe_deflates_by_the_debtor() -> None:
    rows = _member_rows(
        [_row(provider="country/FRA", recipient="country/KEN", year="2020", value="10")]
    )
    exchange = _rates((_USD, 2020, 1.0, "actual"), (_USD, 2015, 1.0, "actual"))
    deflator = _deflators(
        ("country/FRA", 2015, 1.0, "actual"),
        ("country/FRA", 2020, 1.0, "actual"),
        ("country/KEN", 2015, 80.0, "actual"),
        ("country/KEN", 2020, 100.0, "actual"),
    )
    converted, _ = _convert(
        rows,
        target=UniformTarget(_USD),
        exchange=exchange,
        recipe=_IDS,
        deflator=deflator,
        base_year=2015,
    )
    assert converted[VALUE_COLUMN].to_pylist()[0] == repr(10.0 * (80.0 / 100.0) * 1.0 / 1.0)


# ---------------------------------------------------------------------------
# Identity, cross rate, currency keys
# ---------------------------------------------------------------------------


def test_usd_to_usd_is_an_identity() -> None:
    rows = _member_rows([_row(year="2022", value="777")])
    converted, _ = _convert(
        rows, target=UniformTarget(_USD), exchange=_rates((_USD, 2022, 1.0, "actual"))
    )
    assert converted[VALUE_COLUMN].to_pylist()[0] == repr(777.0)


def test_cross_rate_derives_through_the_dollar() -> None:
    rows = _member_rows([_row(year="2022", value="1000")])
    exchange = _rates((_KES, 2022, 110.5, "actual"), (_GBP, 2022, 0.79, "actual"))
    converted, _ = _convert(rows, target=UniformTarget(_GBP), exchange=exchange, native=_KES)
    assert converted[VALUE_COLUMN].to_pylist()[0] == repr(1000.0 * 0.79 / 110.5)


def test_rates_are_read_by_currency_and_never_by_a_country_row() -> None:
    rows = _member_rows([_row(year="2021", value="440")])
    exchange = _rates(
        ("country/DEU", 2021, 0.5, "actual"),  # decoy: a country, not a currency
        (_EUR, 2021, 0.88, "actual"),
        (_USD, 2021, 1.0, "actual"),
    )
    converted, _ = _convert(rows, target=UniformTarget(_USD), exchange=exchange, native=_EUR)
    assert converted[VALUE_COLUMN].to_pylist()[0] == repr(500.0)


def test_role_shaped_target_carries_a_different_currency_per_row() -> None:
    rows = _member_rows(
        [
            _row(recipient="country/KEN", year="2022", value="100"),
            _row(recipient="country/GBR", year="2022", value="200"),
        ]
    )
    exchange = _rates(
        (_USD, 2022, 1.0, "actual"), (_KES, 2022, 112.0, "actual"), (_GBP, 2022, 0.81, "actual")
    )
    target = RoleShapedTarget(
        spec_role="destination", currency_by_entity={"country/KEN": _KES, "country/GBR": _GBP}
    )
    converted, _ = _convert(rows, target=target, exchange=exchange)
    assert converted[VALUE_COLUMN].to_pylist() == [
        repr(100.0 * 112.0 / 1.0),
        repr(200.0 * 0.81 / 1.0),
    ]
    units = converted["unit"].to_pylist()
    assert units[0] == {"currency": "KES", "prices": "current", "base_year": None}
    assert units[1] == {"currency": "GBP", "prices": "current", "base_year": None}


def test_role_shaped_target_raises_usage_error_for_an_unmapped_entity() -> None:
    rows = _member_rows([_row(recipient="country/TZA", year="2022", value="100")])
    target = RoleShapedTarget(spec_role="destination", currency_by_entity={"country/KEN": _KES})
    with pytest.raises(MaterialisationUsageError):
        _convert(rows, target=target, exchange=_rates((_USD, 2022, 1.0, "actual")))


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def test_fallback_substitutes_r_in_the_deflator_terms() -> None:
    rows = _member_rows([_row(provider="country/BEN", year="2020", value="500")])
    exchange = _rates((_USD, 2020, 1.0, "actual"), (_KES, 2015, 140.0, "actual"))
    deflator = _deflators(
        ("DACCountriesTotal", 2015, 90.0, "actual"), ("DACCountriesTotal", 2020, 110.0, "actual")
    )
    converted, record = _convert(
        rows,
        target=UniformTarget(_KES),
        exchange=exchange,
        recipe=_recipe(fallback_entity="DACCountriesTotal"),
        deflator=deflator,
        base_year=2015,
    )
    assert converted[VALUE_COLUMN].to_pylist()[0] == repr(500.0 * (90.0 / 110.0) * 140.0 / 1.0)
    assert record.fallback_count == 1
    assert record.fallback_examples == (("country/BEN", "DACCountriesTotal"),)


def test_the_deployed_dac_recipe_falls_back_to_the_dac_total() -> None:
    assert _DAC.fallback_entity == "DACCountriesTotal"
    assert _IMF.fallback_entity is None
    assert _IDS.fallback_entity is None


def test_null_v_row_is_neither_substituted_nor_counted_as_a_fallback() -> None:
    rows = _member_rows([_row(provider="country/BEN", year="2020", value="")])
    deflator = _deflators(
        ("DACCountriesTotal", 2015, 90.0, "actual"), ("DACCountriesTotal", 2020, 110.0, "actual")
    )
    converted, record = _convert(
        rows,
        target=UniformTarget(_KES),
        exchange=_rates(),
        recipe=_recipe(fallback_entity="DACCountriesTotal"),
        deflator=deflator,
        base_year=2015,
    )
    assert converted[VALUE_COLUMN].to_pylist()[0] == ""
    assert record.fallback_count == 0
    assert record.fallback_examples == ()


def test_r_present_but_missing_a_year_is_a_gap_not_a_fallback() -> None:
    """Benin holds a deflator in 2015 alone, so its 2020 row is left out rather than read
    through the fallback entity."""
    rows = _member_rows([_row(provider="country/BEN", year="2020", value="500")])
    exchange = _rates((_USD, 2020, 1.0, "actual"), (_KES, 2015, 140.0, "actual"))
    deflator = _deflators(("country/BEN", 2015, 90.0, "actual"))  # missing at t=2020
    converted, record = _convert(
        rows,
        target=UniformTarget(_KES),
        exchange=exchange,
        recipe=_recipe(fallback_entity="DACCountriesTotal"),
        deflator=deflator,
        base_year=2015,
    )
    assert converted.num_rows == 0
    assert (record.fallback_count, record.deflator_omitted) == (0, (("country/BEN", (2020,)),))


# ---------------------------------------------------------------------------
# Gap causes
# ---------------------------------------------------------------------------


def test_gap_rate_missing_names_the_currency_and_year() -> None:
    rows = _member_rows([_row(year="2023", value="10")])
    exchange = _rates((_KES, 2023, 120.0, "actual"))  # X[USD, 2023] missing
    with pytest.raises(RateCoverageGapError) as exc_info:
        _convert(rows, target=UniformTarget(_KES), exchange=exchange)
    assert exc_info.value.cause == "rate_missing"
    assert exc_info.value.entities == ("CurrencyUSD@2023",)


def test_a_currency_with_no_rate_rows_refuses_rate_missing() -> None:
    """A withdrawn currency resolves as a `CurrencyUnit` and refuses here."""
    rows = _member_rows([_row(year="2023", value="10")])
    with pytest.raises(RateCoverageGapError) as exc_info:
        _convert(
            rows,
            target=UniformTarget(_USD),
            exchange=_rates((_USD, 2023, 1.0, "actual")),
            native="CurrencyFRF",
        )
    assert exc_info.value.cause == "rate_missing"
    assert exc_info.value.entities == ("CurrencyFRF@2023",)


@pytest.mark.parametrize(
    ("benin_deflators", "kept_years", "omitted"),
    [
        pytest.param(
            [(2010, 80.0), (2019, 95.0)],
            ["2019", "2019", "2020"],
            (("country/BEN", (2020,)),),
            id="a-year-missing-leaves-that-year-out",
        ),
        pytest.param(
            [(2019, 95.0), (2020, 99.0)],
            ["2019", "2020"],
            (("country/BEN", (2019, 2020)),),
            id="the-base-year-missing-leaves-the-entity-out",
        ),
    ],
)
def test_a_missing_deflator_leaves_out_only_its_entitys_rows(
    benin_deflators, kept_years, omitted
) -> None:
    """France holds every deflator; Benin lacks one, in a row's year or in the base year.
    Only Benin's rows that read the missing value are left out, the rows kept convert in
    their order, and the record names Benin with the years it lost."""
    rows = _member_rows(
        [
            _row(year="2019", value="10"),
            _row(provider="country/BEN", year="2019", value="20"),
            _row(year="2020", value="30"),
            _row(provider="country/BEN", year="2020", value="40"),
        ]
    )
    exchange = _rates(
        (_USD, 2019, 1.0, "actual"), (_USD, 2020, 1.0, "actual"), (_KES, 2010, 78.0, "actual")
    )
    deflator = _deflators(
        ("country/FRA", 2010, 88.0, "actual"),
        ("country/FRA", 2019, 98.0, "actual"),
        ("country/FRA", 2020, 99.0, "actual"),
        *(("country/BEN", year, value, "actual") for year, value in benin_deflators),
    )
    converted, record = _convert(
        rows, target=UniformTarget(_KES), exchange=exchange, deflator=deflator, base_year=2010
    )
    assert converted["TIME_PERIOD"].to_pylist() == kept_years
    assert record.deflator_omitted == omitted


def test_every_row_missing_its_deflator_converts_none() -> None:
    """A member whose every row lacks its deflator converts no row and names them all; the
    request decides whether that refuses (`refuse_if_every_row_omitted`)."""
    rows = _member_rows([_row(year="2019", value="250")])
    exchange = _rates((_USD, 2019, 1.0, "actual"), (_KES, 2010, 78.0, "actual"))
    deflator = _deflators(("country/FRA", 2010, 88.0, "actual"))  # D[R, 2019] missing
    converted, record = _convert(
        rows, target=UniformTarget(_KES), exchange=exchange, deflator=deflator, base_year=2010
    )
    assert converted.num_rows == 0
    assert record.deflator_omitted == (("country/FRA", (2019,)),)


def test_a_rate_gap_refuses_even_in_a_row_a_missing_deflator_leaves_out() -> None:
    rows = _member_rows([_row(year="2019", value="250")])
    exchange = _rates((_KES, 2010, 78.0, "actual"))  # X[USD, 2019] missing
    deflator = _deflators(("country/FRA", 2010, 88.0, "actual"))  # D[R, 2019] missing
    with pytest.raises(RateCoverageGapError) as exc_info:
        _convert(
            rows, target=UniformTarget(_KES), exchange=exchange, deflator=deflator, base_year=2010
        )
    assert exc_info.value.cause == "rate_missing"


def test_a_row_with_no_deflator_entity_refuses_rather_than_leaving_it_out() -> None:
    """A null cell on the deflator role names no entity the record could list, so the row
    refuses `deflator_missing` even beside a row that converts."""
    rows = _member_rows(
        [
            _row(year="2019", value="10"),
            {**_row(year="2019", value="20"), "financingProvider": None},
        ]
    )
    exchange = _rates((_USD, 2019, 1.0, "actual"), (_KES, 2010, 78.0, "actual"))
    deflator = _deflators(
        ("country/FRA", 2010, 88.0, "actual"), ("country/FRA", 2019, 98.0, "actual")
    )
    with pytest.raises(RateCoverageGapError) as exc_info:
        _convert(
            rows, target=UniformTarget(_KES), exchange=exchange, deflator=deflator, base_year=2010
        )
    assert (exc_info.value.cause, exc_info.value.entities) == (
        "deflator_missing",
        ("None@2010", "None@2019"),
    )


# ---------------------------------------------------------------------------
# Projections and forecast policy
# ---------------------------------------------------------------------------


def _projection_case(
    *, base_rate_facet: str = "actual", base_deflator_facet: str = "actual", t_facet: str = "actual"
) -> tuple[pa.Table, RateTable, RateTable]:
    rows = _member_rows([_row(year="2019", value="250")])
    exchange = _rates((_USD, 2019, 1.0, t_facet), (_KES, 2010, 78.0, base_rate_facet))
    deflator = _deflators(
        ("country/FRA", 2010, 88.0, base_deflator_facet), ("country/FRA", 2019, 108.0, "actual")
    )
    return rows, exchange, deflator


_PROJECTION_EXPECTED = 250.0 * (88.0 / 108.0) * 78.0 / 1.0


@pytest.mark.parametrize(
    ("facets", "key"),
    [
        ({"base_deflator_facet": "projected"}, "country/FRA@2010"),
        ({"base_rate_facet": "projected"}, "CurrencyKES@2010"),
    ],
)
def test_flag_projected_converts_a_projected_base_year_term_and_flags_it(
    facets: dict[str, str], key: str
) -> None:
    rows, exchange, deflator = _projection_case(**facets)
    converted, record = _convert(
        rows,
        target=UniformTarget(_KES),
        exchange=exchange,
        recipe=_recipe(forecast_policy="FlagProjected"),
        deflator=deflator,
        base_year=2010,
    )
    assert converted[VALUE_COLUMN].to_pylist()[0] == repr(_PROJECTION_EXPECTED)
    assert record.projected_count == 1
    assert record.projected_examples == (key,)


def test_refuse_projected_refuses_a_projected_base_year_term() -> None:
    rows, exchange, deflator = _projection_case(base_deflator_facet="projected")
    with pytest.raises(RateCoverageGapError) as exc_info:
        _convert(
            rows,
            target=UniformTarget(_KES),
            exchange=exchange,
            recipe=_recipe(forecast_policy="RefuseProjected"),
            deflator=deflator,
            base_year=2010,
        )
    assert exc_info.value.cause == "base_year_projected"
    assert exc_info.value.entities == ("country/FRA@2010",)


def test_flag_projected_converts_and_marks_a_year_t_projection() -> None:
    rows, exchange, deflator = _projection_case(t_facet="projected")
    converted, record = _convert(
        rows, target=UniformTarget(_KES), exchange=exchange, deflator=deflator, base_year=2010
    )
    assert converted[VALUE_COLUMN].to_pylist()[0] == repr(_PROJECTION_EXPECTED)
    assert record.projected_count == 1
    assert record.projected_examples == ("CurrencyUSD@2019",)


def test_a_row_with_two_projected_terms_counts_once_and_names_both() -> None:
    rows, exchange, deflator = _projection_case(
        base_deflator_facet="projected", t_facet="projected"
    )
    _, record = _convert(
        rows, target=UniformTarget(_KES), exchange=exchange, deflator=deflator, base_year=2010
    )
    assert record.projected_count == 1
    assert record.projected_examples == ("CurrencyUSD@2019", "country/FRA@2010")


def test_refuse_projected_treats_a_year_t_projection_as_a_gap() -> None:
    rows, exchange, deflator = _projection_case(t_facet="projected")
    with pytest.raises(RateCoverageGapError) as exc_info:
        _convert(
            rows,
            target=UniformTarget(_KES),
            exchange=exchange,
            recipe=_recipe(forecast_policy="RefuseProjected"),
            deflator=deflator,
            base_year=2010,
        )
    assert exc_info.value.cause == "rate_missing"
    assert exc_info.value.entities == ("CurrencyUSD@2019",)


def test_base_year_projected_outranks_rate_missing() -> None:
    rows = _member_rows([_row(year="2019", value="250")])
    exchange = _rates((_KES, 2010, 78.0, "actual"))  # X[USD, 2019] missing
    deflator = _deflators(
        ("country/FRA", 2010, 88.0, "projected"), ("country/FRA", 2019, 108.0, "actual")
    )
    with pytest.raises(RateCoverageGapError) as exc_info:
        _convert(
            rows,
            target=UniformTarget(_KES),
            exchange=exchange,
            recipe=_recipe(forecast_policy="RefuseProjected"),
            deflator=deflator,
            base_year=2010,
        )
    assert exc_info.value.cause == "base_year_projected"


# ---------------------------------------------------------------------------
# Null V
# ---------------------------------------------------------------------------


def test_null_native_value_stays_null_and_is_not_a_gap() -> None:
    rows = _member_rows([_row(year="2099", value="")])  # no rate exists for 2099 at all
    converted, record = _convert(
        rows, target=UniformTarget(_KES), exchange=_rates((_USD, 2000, 1.0, "actual"))
    )
    assert converted[VALUE_COLUMN].to_pylist()[0] == ""
    assert record.fallback_count == 0
    assert record.projected_count == 0


def test_null_native_value_on_a_role_shaped_target_skips_the_currency_lookup() -> None:
    rows = _member_rows(
        [
            _row(recipient="country/TZA", year="2099", value=""),
            _row(recipient="country/KEN", year="2022", value="100"),
        ]
    )
    exchange = _rates((_USD, 2022, 1.0, "actual"), (_KES, 2022, 112.0, "actual"))
    target = RoleShapedTarget(spec_role="destination", currency_by_entity={"country/KEN": _KES})
    converted, record = _convert(rows, target=target, exchange=exchange)
    assert converted[VALUE_COLUMN].to_pylist() == ["", repr(100.0 * 112.0 / 1.0)]
    units = converted["unit"].to_pylist()
    assert units[0] is None
    assert units[1] == {"currency": "KES", "prices": "current", "base_year": None}
    assert record.fallback_count == 0
    assert record.projected_count == 0


# ---------------------------------------------------------------------------
# Unrecognized recipe fields, and caller-usage misuse
# ---------------------------------------------------------------------------


def test_unrecognized_forecast_policy_raises_upstream_response_error() -> None:
    rows = _member_rows([_row(year="2021", value="400")])
    with pytest.raises(UpstreamResponseError):
        _convert(
            rows,
            target=UniformTarget(_KES),
            exchange=_rates((_USD, 2021, 1.0, "actual")),
            recipe=_recipe(forecast_policy="SomethingElse"),
        )


@pytest.mark.parametrize(("deflator_table_present", "base_year"), [(False, 2010), (True, None)])
def test_constant_prices_requires_deflator_table_and_base_year(
    deflator_table_present: bool, base_year: int | None
) -> None:
    rows = _member_rows([_row(year="2019", value="250")])
    with pytest.raises(MaterialisationUsageError):
        convert_member_marked(
            rows=rows,
            unit_request=ResolvedUnitRequest(
                prices="constant", target=UniformTarget(_KES), base_year=base_year
            ),
            recipe=_recipe(),
            native_currency=_USD,
            native_unit_ref=_USD,
            declared_roles=_ROLES,
            exchange_rate_table=_rates((_USD, 2019, 1.0, "actual")),
            deflator_table=_deflators() if deflator_table_present else None,
        )


# ---------------------------------------------------------------------------
# Bit-for-bit replay: hand-picked adversarial values
# ---------------------------------------------------------------------------


def test_convert_scalar_current_is_left_to_right_not_reassociated() -> None:
    v, x_target_t, x_source_t = -489861948521.1566, -9129825816.118042, -101017870422.5238
    pinned = convert_scalar(v, prices="current", x_target_t=x_target_t, x_source_t=x_source_t)
    reassociated = v * (x_target_t / x_source_t)
    assert pinned == (v * x_target_t) / x_source_t
    assert pinned != reassociated  # confirms this triple really is order-sensitive


def test_convert_scalar_constant_is_left_to_right_not_reassociated() -> None:
    v, x_target_b, x_source_t = -566801205738.7732, -155766848834.56543, -941918424850.2642
    pinned = convert_scalar(
        v,
        prices="constant",
        d_r_b=5.0,
        d_r_t=5.0,  # deflator ratio == 1.0 exactly, isolating the outer multiply/divide chain
        x_target_b=x_target_b,
        x_source_t=x_source_t,
    )
    reassociated = v * (x_target_b / x_source_t)
    assert pinned == (v * x_target_b) / x_source_t
    assert pinned != reassociated


# ---------------------------------------------------------------------------
# Bit-for-bit replay: a >=10,000-row random sweep per price basis
# ---------------------------------------------------------------------------

_SWEEP_ROWS = 10_000
_SWEEP_DISTINCT_YEARS = 50
_EXTREME_SWEEP_VALUES: tuple[float, ...] = (-0.0, 0.0, 1e150, -1e150, 1e-150, -1e-150)


def _sweep_years() -> list[int]:
    return [2000 + year for year in range(_SWEEP_DISTINCT_YEARS)]


def _sweep_values(rng: random.Random, count: int) -> list[float]:
    """`count` native `V` values: hand-picked extremes (negative zero, very
    large and very small magnitudes) plus uniformly random fill."""
    values = list(_EXTREME_SWEEP_VALUES)
    values += [rng.uniform(-1e9, 1e9) for _ in range(count - len(values))]
    return values


def _current_sweep(rng: random.Random) -> tuple[pa.Table, list[float]]:
    target_rate_by_year = {year: rng.uniform(0.01, 1000.0) for year in _sweep_years()}
    source_rate_by_year = {year: rng.uniform(0.01, 1000.0) for year in _sweep_years()}
    records: list[dict[str, str | None]] = []
    expected: list[float] = []
    for i, v in enumerate(_sweep_values(rng, _SWEEP_ROWS)):
        year = 2000 + (i % _SWEEP_DISTINCT_YEARS)
        records.append(_row(year=str(year), value=repr(v)))
        expected.append(
            convert_scalar(
                v,
                prices="current",
                x_target_t=target_rate_by_year[year],
                x_source_t=source_rate_by_year[year],
            )
        )
    exchange = _rates(
        *((_GBP, year, rate, "actual") for year, rate in target_rate_by_year.items()),
        *((_KES, year, rate, "actual") for year, rate in source_rate_by_year.items()),
    )
    converted, _ = _convert(
        _member_rows(records), target=UniformTarget(_GBP), exchange=exchange, native=_KES
    )
    return converted, expected


def _constant_sweep(rng: random.Random) -> tuple[pa.Table, list[float]]:
    b = 1995
    d_r_t_by_year = {year: rng.uniform(0.01, 1000.0) for year in _sweep_years()}
    x_source_t_by_year = {year: rng.uniform(0.01, 1000.0) for year in _sweep_years()}
    d_r_b = rng.uniform(0.01, 1000.0)
    x_target_b = rng.uniform(0.01, 1000.0)
    records: list[dict[str, str | None]] = []
    expected: list[float] = []
    for i, v in enumerate(_sweep_values(rng, _SWEEP_ROWS)):
        year = 2000 + (i % _SWEEP_DISTINCT_YEARS)
        records.append(_row(year=str(year), value=repr(v)))
        expected.append(
            convert_scalar(
                v,
                prices="constant",
                d_r_b=d_r_b,
                d_r_t=d_r_t_by_year[year],
                x_target_b=x_target_b,
                x_source_t=x_source_t_by_year[year],
            )
        )
    exchange = _rates(
        *((_KES, year, rate, "actual") for year, rate in x_source_t_by_year.items()),
        (_GBP, b, x_target_b, "actual"),
    )
    deflator = _deflators(
        ("country/FRA", b, d_r_b, "actual"),
        *(("country/FRA", year, rate, "actual") for year, rate in d_r_t_by_year.items()),
    )
    converted, _ = _convert(
        _member_rows(records),
        target=UniformTarget(_GBP),
        exchange=exchange,
        native=_KES,
        deflator=deflator,
        base_year=b,
    )
    return converted, expected


def _assert_bit_exact(actual: Sequence[float], expected: Sequence[float]) -> None:
    assert len(actual) == _SWEEP_ROWS
    for one_actual, one_expected in zip(actual, expected, strict=True):
        assert _bit_exact(one_actual, one_expected)


@pytest.mark.parametrize("sweep", [_current_sweep, _constant_sweep])
def test_bit_for_bit_sweep(sweep) -> None:
    converted, expected = sweep(random.Random(20260914))
    actual = [float(cast(str, cell)) for cell in converted[VALUE_COLUMN].to_pylist()]
    _assert_bit_exact(actual, expected)


@pytest.mark.parametrize("sweep", [_current_sweep, _constant_sweep])
def test_bit_for_bit_sweep_matches_columnar_observed_values(sweep) -> None:
    """`columnar.observed_values`, the parser `frame_build.py` runs on a
    converted `OBS_VALUE` column, agrees bit for bit with `convert_scalar`."""
    converted, expected = sweep(random.Random(20260918))
    actual = cast("list[float]", observed_values(converted[VALUE_COLUMN]).to_pylist())
    _assert_bit_exact(actual, expected)


def _part_record(**changes: object) -> ConversionRecord:
    record = ConversionRecord(
        recipe_id="ConversionRecipeDAC",
        exchange_rate_table_version="rates-v1",
        deflator_table_version="deflators-v1",
        native_currency=_USD,
        native_unit_ref=_USD,
        table_provenance_id="provenance/rates",
        table_refresh_date="2026-10-02",
        fallback_count=1,
        fallback_examples=(("KEN", "2020"),),
        projected_count=1,
        projected_examples=("KEN@2025",),
    )
    return dataclasses.replace(record, **changes)  # type: ignore[arg-type]


def test_merged_record_sums_parts_under_one_recipe_and_table_versions() -> None:
    merged = merged_record(
        [
            _part_record(),
            _part_record(fallback_examples=(("SEN", "2021"),), projected_examples=("SEN@2025",)),
        ]
    )
    assert merged is not None
    assert merged.fallback_count == 2
    assert merged.fallback_examples == (("KEN", "2020"), ("SEN", "2021"))
    assert merged.projected_count == 2
    assert merged.projected_examples == ("KEN@2025", "SEN@2025")


@pytest.mark.parametrize(
    "changes",
    [
        {"recipe_id": "ConversionRecipeIDS"},
        {"exchange_rate_table_version": "rates-v2"},
        {"deflator_table_version": "deflators-v2"},
        {"deflator_table_version": None},
    ],
)
def test_merged_record_refuses_parts_under_different_trace_keys(
    changes: dict[str, object],
) -> None:
    with pytest.raises(MaterialisationUsageError, match="one recipe and table-version pair"):
        merged_record([_part_record(), _part_record(**changes)])
