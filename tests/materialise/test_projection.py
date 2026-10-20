"""Tests the long-output layout `materialise.projection` puts on a reshaped table."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from dfo_frame_grammar.grammar import ComposedUnit, StatVarPart, UnitRequest

from code_index.types import PartyClass, PartyKind, PartyRow, WholeMarker
from materialise.errors import MaterialisationUsageError
from materialise.projection import (
    NativeCurrency,
    UnitNaming,
    UnitParts,
    long_layout_default_enrichments,
    long_output_selection,
    period_cell,
    period_grain,
    plan_long_projection,
    split_totals,
    split_unit,
    totals_roles,
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


_CLASS_OF_KIND: Mapping[PartyKind, PartyClass | None] = {
    "party": "official",
    "total": None,
    "residual": "residual",
}


def _parties(kinds: Mapping[str, PartyKind]) -> dict[str, PartyRow]:
    return {
        ref: PartyRow(dcid=ref, kind=kind, party_class=_CLASS_OF_KIND[kind])
        for ref, kind in kinds.items()
    }


_PARTY_KINDS = _parties({"country/USA": "party", "country/KEN": "party"})

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


def _project(
    source: Table = _SOURCE,
    *,
    names=_NAMES,
    parties=_PARTY_KINDS,
    axis_codes=_AXIS_CODES,
    entity_columns=("financingProvider", "financingRecipient"),
):
    return plan_long_projection(
        source.columns,
        entity_columns=entity_columns,
        axis_codes=axis_codes,
        unit_naming=_UNIT_NAMING,
        grain=period_grain(_periods(source)),
    ).apply(source, names=names, parties=parties)


def _periods(source: Table) -> tuple[object, ...]:
    if "time_period" not in source.columns:
        return ()
    at = source.columns.index("time_period")
    return tuple(row[at] for row in source.rows)


def _row(table: Table, index: int) -> dict[str, object]:
    return dict(zip(table.columns, table.rows[index], strict=True))


def test_a_long_table_takes_the_canonical_column_order_and_drops_status():
    projected = _project()

    assert projected.table.columns == (
        "stat_var",
        "financingProvider",
        "financingProvider_label",
        "financingProvider_kind",
        "financingRecipient",
        "financingRecipient_label",
        "financingRecipient_kind",
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
        "official",
        "country/KEN",
        "Kenya",
        "country or organisation",
        "KEN",
        "dcid:CRSPurposeCode122",
        "Basic health care",
        "dcid:CRSTypeOfFinanceCode110",
        "Standard grant",
        2022,
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


def test_a_value_without_a_name_or_kind_leaves_its_cell_empty_and_counts_a_miss():
    projected = _project(
        names={"country/USA": "United States", "p1": "Source one"},
        parties=_parties({"country/USA": "total"}),
    )

    assert [
        (row["financingRecipient_label"], row["financingRecipient_kind"])
        for row in (_row(projected.table, i) for i in (0, 1))
    ] == [(None, None), (None, None)]
    assert _row(projected.table, 0)["financingProvider_kind"] == "total"
    # `m1` has no name either, and an unnamed method is not a miss.
    assert projected.unavailable == {
        "financingRecipient_label": 2,
        "financingRecipient_kind": 2,
    }


def test_a_party_role_gets_one_kind_column_in_words_and_a_subject_role_none():
    # The long layout reads the party cells itself; the render drops every party-field token
    # before reshape, so no machine party column reaches a long table.
    source = Table(
        columns=("observationAbout", "financingRecipient", "value"),
        rows=(("country/USA", "country/KEN", 1.0),),
    )

    projected = _project(
        source,
        entity_columns=("observationAbout", "financingRecipient"),
        parties=_parties({"country/KEN": "party"}),
    )

    assert projected.table.columns == (
        "observationAbout",
        "observationAbout_label",
        "financingRecipient",
        "financingRecipient_label",
        "financingRecipient_kind",
        "value",
    )
    assert projected.table.rows == (
        ("country/USA", "United States", "country/KEN", "Kenya", "country or organisation", 1.0),
    )
    assert not {
        f"{role}_{field}"
        for role in ("observationAbout", "financingRecipient")
        for field in ("party_kind", "party_class", "party_whole")
    } & set(projected.table.columns)


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
        grain="year",
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
        pytest.param(
            ComposedUnit(numerator="USDollar", denominator=StatVarPart("dcid:Count_Person")),
            "current",
            UnitParts(
                unit='{"denominator":{"stat_var":"dcid:Count_Person"},"numerator":"USDollar"}',
                currency=None,
                price_basis=None,
                base_year=None,
            ),
            id="composed",
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


def test_the_default_enrichments_are_every_roles_name_and_each_party_roles_kind():
    assert long_layout_default_enrichments(
        ("observationAbout", "financingProvider", "creditor")
    ) == (
        "observationAbout_label",
        "financingProvider_label",
        "financingProvider_kind",
        "creditor_label",
        "creditor_kind",
    )


def test_a_layout_refuses_a_table_with_other_columns():
    projection = plan_long_projection(
        ("stat_var", "value"),
        entity_columns=(),
        axis_codes={},
        unit_naming=_UNIT_NAMING,
        grain=None,
    )

    with pytest.raises(MaterialisationUsageError):
        projection.apply(Table(columns=("value", "stat_var"), rows=()), names={}, parties={})


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


@pytest.mark.parametrize(
    ("periods", "grain", "written"),
    [
        pytest.param(("2022", "2023"), "year", (2022, 2023), id="year-points-are-integers"),
        pytest.param(("2022-03",), "month", ("2022-03",), id="a-month-stays-text"),
        pytest.param(("2019/2021",), None, ("2019/2021",), id="a-span-stays-text"),
        pytest.param(("2022", "2019/2021"), None, ("2022", "2019/2021"), id="a-later-span"),
        pytest.param(("2019/2021", "2022"), None, ("2019/2021", "2022"), id="an-earlier-span"),
        pytest.param((), None, (), id="no-rows"),
    ],
)
def test_year_points_are_written_as_integers(periods, grain, written):
    source = Table(columns=("time_period", "value"), rows=tuple((p, 1.0) for p in periods))

    assert period_grain(periods) == grain
    assert tuple(period_cell(grain, period) for period in periods) == written
    assert tuple(row[0] for row in _project(source, entity_columns=()).table.rows) == written


@pytest.mark.parametrize(
    ("identity", "expected"),
    [
        pytest.param(
            {"entities": {"origin": ["country/GBR"]}, "open_roles": ["destination"]},
            ("financingRecipient",),
            id="an-open-role",
        ),
        pytest.param(
            {"entities": {"origin": ["country/GBR"], "destination": ["DACDevelopingCountries"]}},
            (),
            id="a-role-bound-to-one-total-sets-nothing-apart",
        ),
        pytest.param(
            {"entities": {"recipient": ["country/KEN", "africa"]}},
            ("financingRecipient",),
            id="a-role-bound-to-several-in-the-callers-spelling",
        ),
        pytest.param(
            {"entities": {"observationAbout": ["country/KEN", "country/UGA"]}},
            (),
            id="a-subject-role-is-no-party-role",
        ),
        pytest.param({"entities": {}}, (), id="a-role-asked-as-its-total"),
    ],
)
def test_totals_come_from_roles_the_identity_leaves_open_or_binds_to_several(identity, expected):
    entity_columns = ("observationAbout", "financingProvider", "financingRecipient")

    assert totals_roles(identity, entity_columns) == expected


def test_a_row_is_a_total_when_a_split_role_holds_a_total_and_residuals_stay():
    columns = ("financingProvider_kind", "financingRecipient_kind", "value")
    table = Table(
        columns=columns,
        rows=(
            ("official", "country or organisation", 1.0),
            ("official", "unallocated", 2.0),
            ("official", "total", 3.0),
            ("total", "country or organisation", 4.0),
            ("private", "country or organisation", 5.0),
            ("official", "whole total", 6.0),
            ("whole total", "country or organisation", 7.0),
        ),
    )

    parties, totals = split_totals(table, roles=("financingRecipient", "observationAbout"))

    assert (parties.columns, totals.columns) == (columns, columns)
    assert [row[2] for row in parties.rows] == [1.0, 2.0, 4.0, 5.0, 7.0]
    assert [row[2] for row in totals.rows] == [3.0, 6.0]


_DAC2A = "provenance/OECDDAC2A"
_DAC2B = "provenance/OECDDAC2B"
_WHOLE_PARTIES = {
    "DACAllRecipients": PartyRow(
        dcid="DACAllRecipients",
        kind="total",
        datasets=(_DAC2A, _DAC2B),
        axes=("destination",),
        whole=(WholeMarker(dataset=_DAC2A, covers=("official", "residual"), years=(1984, 2024)),),
    ),
    "DACOfficialDonors": PartyRow(
        dcid="DACOfficialDonors",
        kind="total",
        party_class="official",
        datasets=(_DAC2A, _DAC2B),
        axes=("origin",),
        whole=(
            WholeMarker(dataset=_DAC2A, covers=("official",)),
            WholeMarker(dataset=_DAC2B, covers=("official",)),
        ),
    ),
}


def _kinds(
    columns: tuple[str, ...], rows: tuple[tuple[object, ...], ...], *, role: str
) -> list[object]:
    table = Table(columns=columns, rows=rows)
    projected = plan_long_projection(
        table.columns,
        entity_columns=(role,),
        axis_codes={},
        unit_naming=_UNIT_NAMING,
        grain=None,
    ).apply(table, names={}, parties=_WHOLE_PARTIES)
    return [_row(projected.table, i)[f"{role}_kind"] for i in range(len(rows))]


_RECIPIENT_COLUMNS = ("stat_var", "financingRecipient", "time_period", "provenance", "value")


def test_one_frame_reads_an_all_recipients_total_by_dataset_and_year():
    """DAC2A holds the total from 1984; DAC2B carries no marker, so every DAC2B row reads
    `total`."""
    rows = (
        ("dcid:SV_A", "DACAllRecipients", "2021", _DAC2A, 1.0),
        ("dcid:SV_A", "DACAllRecipients", "1984", _DAC2A, 2.0),
        ("dcid:SV_A", "DACAllRecipients", "1983", _DAC2A, 3.0),
        ("dcid:SV_A", "DACAllRecipients", "2021", _DAC2B, 4.0),
        ("dcid:SV_A", "DACAllRecipients", "1990", _DAC2B, 5.0),
    )

    assert _kinds(_RECIPIENT_COLUMNS, rows, role="financingRecipient") == [
        "whole total",
        "whole total",
        "total",
        "total",
        "total",
    ]


def test_official_donors_read_whole_total_in_both_datasets():
    columns = ("stat_var", "financingProvider", "time_period", "provenance", "value")
    rows = (
        ("dcid:SV_A", "DACOfficialDonors", "2021", _DAC2A, 1.0),
        ("dcid:SV_A", "DACOfficialDonors", "2021", _DAC2B, 2.0),
    )

    assert _kinds(columns, rows, role="financingProvider") == ["whole total", "whole total"]


@pytest.mark.parametrize(
    ("columns", "rows"),
    [
        pytest.param(
            ("stat_var", "financingRecipient", "provenance", "value"),
            (("dcid:SV_A", "DACAllRecipients", _DAC2A, 1.0),),
            id="no-period-column",
        ),
        pytest.param(
            ("stat_var", "financingRecipient", "time_period", "value"),
            (("dcid:SV_A", "DACAllRecipients", "2021", 1.0),),
            id="no-provenance-column",
        ),
        pytest.param(
            _RECIPIENT_COLUMNS,
            (("dcid:SV_A", "DACAllRecipients", "2021", None, 1.0),),
            id="empty-provenance-cell",
        ),
        pytest.param(
            _RECIPIENT_COLUMNS,
            (("dcid:SV_A", "DACAllRecipients", "2021/2022", _DAC2A, 1.0),),
            id="period-across-years",
        ),
    ],
)
def test_a_row_without_a_readable_dataset_or_year_reads_total(columns, rows):
    assert _kinds(columns, rows, role="financingRecipient") == ["total"]


_SIDE_PARTIES = {
    "country/FRA": PartyRow(dcid="country/FRA", kind="party", party_class="official"),
    "GatesFoundation": PartyRow(dcid="GatesFoundation", kind="party", party_class="private"),
    "DACAfricaUnallocated": PartyRow(
        dcid="DACAfricaUnallocated", kind="residual", party_class="residual"
    ),
    **_WHOLE_PARTIES,
}


@pytest.mark.parametrize(
    ("role", "ref", "provenance", "expected"),
    [
        pytest.param(
            "financingProvider", "country/FRA", _DAC2A, "official", id="official-provider"
        ),
        pytest.param(
            "financingProvider", "GatesFoundation", _DAC2A, "private", id="private-provider"
        ),
        pytest.param("creditor", "country/FRA", _DAC2A, "official", id="official-creditor"),
        pytest.param(
            "financingRecipient",
            "country/FRA",
            _DAC2A,
            "country or organisation",
            id="official-recipient",
        ),
        pytest.param(
            "debtor", "country/FRA", _DAC2A, "country or organisation", id="official-debtor"
        ),
        pytest.param(
            "financingRecipient", "GatesFoundation", _DAC2A, "private", id="private-recipient"
        ),
        pytest.param(
            "financingRecipient",
            "DACAfricaUnallocated",
            _DAC2A,
            "unallocated",
            id="residual",
        ),
        pytest.param(
            "financingProvider", "DACOfficialDonors", _DAC2A, "whole total", id="whole-total"
        ),
        pytest.param(
            "financingRecipient", "DACAllRecipients", _DAC2B, "total", id="total-not-whole-here"
        ),
        pytest.param("financingRecipient", "country/NOWHERE", _DAC2A, None, id="not-listed"),
    ],
)
def test_a_long_kind_cell_reads_the_partys_kind_in_words_for_its_side(
    role, ref, provenance, expected
):
    columns = ("stat_var", role, "time_period", "provenance", "value")

    projected = plan_long_projection(
        columns, entity_columns=(role,), axis_codes={}, unit_naming=_UNIT_NAMING, grain=None
    ).apply(
        Table(columns=columns, rows=(("dcid:SV_A", ref, "2021", provenance, 1.0),)),
        names={},
        parties=_SIDE_PARTIES,
    )

    assert _row(projected.table, 0)[f"{role}_kind"] == expected
    assert not any("_party_" in column for column in projected.table.columns)
