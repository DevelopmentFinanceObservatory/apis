"""Tests the long-output layout `materialise.projection` puts on a reshaped table."""

from __future__ import annotations

import pytest
from dfo_frame_grammar.grammar import UnitRequest

from materialise.errors import MaterialisationUsageError
from materialise.projection import (
    NativeCurrency,
    UnitNaming,
    UnitParts,
    long_output_selection,
    plan_long_projection,
    split_unit,
)
from materialise.reshape import Table

_AXIS_CODES = {
    "crsPurpose": {"Basic health care": "dcid:CRSPurposeCode122"},
    "crsTypeOfFinance": {"Standard grant": "dcid:CRSTypeOfFinanceCode110"},
}


def _currency_of(unit: str) -> NativeCurrency | None:
    return NativeCurrency(unit.removeprefix("Currency")) if unit.startswith("Currency") else None


_UNIT_NAMING = UnitNaming(
    currency_of=_currency_of,
    native_price_basis=lambda stat_var: "current" if stat_var == "dcid:SV_A" else None,
    label=lambda parts, name: (
        f"{parts.price_basis} {parts.currency}" if parts.currency is not None else name
    ),
)

_NAMES = {
    "country/USA": "United States",
    "country/KEN": "Kenya",
    "m1": "Method one",
    "p1": "Source one",
}

# Reshape's own order: frame columns, then decoded axes, then enrichments.
_SOURCE = Table(
    columns=(
        "stat_var",
        "financingProvider",
        "financingRecipient",
        "time_period",
        "value",
        "unit",
        "measurementMethod",
        "provenance",
        "status",
        "stat_var_crsTypeOfFinance",
        "stat_var_crsPurpose",
        "financingRecipient_iso3",
    ),
    rows=(
        (
            "dcid:SV_A",
            "country/USA",
            "country/KEN",
            "2022",
            10.0,
            "CurrencyUSD",
            None,
            "p1",
            None,
            "Standard grant",
            "Basic health care",
            "KEN",
        ),
        (
            "dcid:SV_B",
            "country/USA",
            "country/KEN",
            "2022",
            5.0,
            UnitRequest(currency="EUR", prices="constant", base_year=2023),
            "m1",
            "p1",
            None,
            "not_declared",
            "",
            "KEN",
        ),
    ),
)


def _project(source: Table = _SOURCE, *, names=_NAMES, axis_codes=_AXIS_CODES):
    return plan_long_projection(
        source.columns,
        entity_columns=("financingProvider", "financingRecipient"),
        axis_codes=axis_codes,
        unit_naming=_UNIT_NAMING,
    ).apply(source, names=names)


def _row(table: Table, index: int) -> dict[str, object]:
    return dict(zip(table.columns, table.rows[index], strict=True))


def test_a_long_table_takes_the_canonical_column_order_and_drops_status():
    projected = _project()

    assert projected.table.columns == (
        "stat_var",
        "financingProvider",
        "financingProvider_label",
        "financingRecipient",
        "financingRecipient_label",
        "financingRecipient_iso3",
        "crsPurpose",
        "crsPurpose_label",
        "crsTypeOfFinance",
        "crsTypeOfFinance_label",
        "time_period",
        "value",
        "unit",
        "unit_label",
        "currency",
        "price_basis",
        "base_year",
        "measurementMethod",
        "measurementMethod_label",
        "provenance",
        "provenance_label",
    )
    assert projected.table.rows[0] == (
        "dcid:SV_A",
        "country/USA",
        "United States",
        "country/KEN",
        "Kenya",
        "KEN",
        "dcid:CRSPurposeCode122",
        "Basic health care",
        "dcid:CRSTypeOfFinanceCode110",
        "Standard grant",
        "2022",
        10.0,
        "CurrencyUSD",
        "current USD",
        "USD",
        "current",
        None,
        None,
        None,
        "p1",
        "Source one",
    )
    assert projected.unavailable == {}


def test_a_converted_row_splits_its_unit_request():
    row = _row(_project().table, 1)

    assert (
        row["unit"],
        row["unit_label"],
        row["currency"],
        row["price_basis"],
        row["base_year"],
    ) == ("CurrencyEUR", "constant EUR", "EUR", "constant", 2023)


def test_a_declared_absence_or_a_decode_gap_leaves_the_code_cell_empty_without_a_miss():
    projected = _project()
    row = _row(projected.table, 1)

    assert (row["crsTypeOfFinance"], row["crsTypeOfFinance_label"]) == (None, "not_declared")
    assert (row["crsPurpose"], row["crsPurpose_label"]) == (None, "")
    assert projected.unavailable == {}


def test_a_label_two_codes_share_leaves_the_code_cell_empty_and_counts_a_miss():
    shared = {**_AXIS_CODES, "crsTypeOfFinance": {"Standard grant": None}}

    projected = _project(axis_codes=shared)

    assert _row(projected.table, 0)["crsTypeOfFinance"] is None
    assert projected.unavailable == {"crsTypeOfFinance": 1}


def test_a_value_without_a_name_leaves_the_name_cell_empty_and_counts_a_miss():
    projected = _project(names={"country/USA": "United States", "p1": "Source one"})

    assert [_row(projected.table, i)["financingRecipient_label"] for i in (0, 1)] == [None, None]
    # `m1` has no name either, and an unnamed method is not a miss.
    assert projected.unavailable == {"financingRecipient_label": 2}


def test_stat_var_label_follows_stat_var_only_when_no_axis_is_decoded():
    source = Table(columns=("stat_var", "value", "unit"), rows=(("dcid:SV_A", 1.0, "Percent"),))

    projected = _project(source, names={"dcid:SV_A": "Aid", "Percent": "Percent"})

    assert projected.table.columns[:3] == ("stat_var", "stat_var_label", "value")
    assert projected.table.rows[0][:2] == ("dcid:SV_A", "Aid")
    assert _row(projected.table, 0)["unit_label"] == "Percent"


def test_the_values_to_name_are_the_name_columns_distinct_non_empty_cells():
    projection = plan_long_projection(
        _SOURCE.columns,
        entity_columns=("financingProvider", "financingRecipient"),
        axis_codes=_AXIS_CODES,
        unit_naming=_UNIT_NAMING,
    )

    # Axes decode, so `stat_var` has no name column. A converted unit cell names no node.
    assert set(projection.refs_to_name(_SOURCE)) == {
        "country/USA",
        "country/KEN",
        "CurrencyUSD",
        "m1",
        "p1",
    }


@pytest.mark.parametrize(
    ("cell", "native_price_basis", "expected"),
    [
        pytest.param(
            "CurrencyKES",
            "current",
            UnitParts(unit="CurrencyKES", currency="KES", price_basis="current", base_year=None),
            id="native-currency",
        ),
        pytest.param(
            "Percent",
            "current",
            UnitParts(unit="Percent", currency=None, price_basis=None, base_year=None),
            id="native-non-currency",
        ),
        pytest.param(
            UnitRequest(currency="USD", prices="constant", base_year=2023),
            None,
            UnitParts(unit="CurrencyUSD", currency="USD", price_basis="constant", base_year=2023),
            id="converted-constant",
        ),
        pytest.param(
            UnitRequest(currency="KES", prices="current"),
            None,
            UnitParts(unit="CurrencyKES", currency="KES", price_basis="current", base_year=None),
            id="converted-current",
        ),
        pytest.param(None, "current", None, id="empty"),
    ],
)
def test_a_unit_cell_splits_into_its_columns(cell, native_price_basis, expected):
    assert (
        split_unit(cell, currency_of=_currency_of, native_price_basis=native_price_basis)
        == expected
    )


def test_a_column_outside_the_layout_follows_it_in_table_order():
    source = Table(columns=("extra_b", "value", "stat_var", "extra_a"), rows=((2, 1.0, "s", 1),))

    projected = _project(source, names={"s": "S"})

    assert projected.table.columns == ("stat_var", "stat_var_label", "value", "extra_b", "extra_a")
    assert projected.table.rows == (("s", "S", 1.0, 2, 1),)


def test_a_layout_refuses_a_table_with_other_columns():
    projection = plan_long_projection(
        ("stat_var", "value"), entity_columns=(), axis_codes={}, unit_naming=_UNIT_NAMING
    )

    with pytest.raises(MaterialisationUsageError):
        projection.apply(Table(columns=("value", "stat_var"), rows=()), names={})


@pytest.mark.parametrize(
    ("select", "expected"),
    [
        pytest.param(None, None, id="no-selection"),
        pytest.param(
            ("stat_var", "value"),
            ("stat_var", "value", "stat_var_crsPurpose", "stat_var_crsTypeOfFinance"),
            id="names-no-axis-keeps-every-axis",
        ),
        pytest.param(
            ("stat_var", "stat_var_crsPurpose"),
            ("stat_var", "stat_var_crsPurpose"),
            id="names-an-axis-keeps-those-named",
        ),
    ],
)
def test_a_selection_keeps_the_axes_stat_var_brings(select, expected):
    assert (
        long_output_selection(
            select, axis_columns=("stat_var_crsPurpose", "stat_var_crsTypeOfFinance")
        )
        == expected
    )
