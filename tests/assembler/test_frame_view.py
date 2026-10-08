"""`frame_view.compute_structural_view`: dimension derivation (stat_var +
entity roles + the three facet columns), distinct-value listing and
truncation, `plural`, period grain from time_period lexical form, coverage,
and series count.
"""

from __future__ import annotations

import pytest

from assembler.errors import BindingInvalid
from assembler.frame_view import compute_structural_view, dimension_columns, hashable_cell
from assembler.testing import build_frame_doc


def _rows(*, periods, recipient="country/ETH"):
    return [
        {
            "stat_var": "dcid:X",
            "recipient": recipient,
            "time_period": period,
            "value": 1.0,
            "unit": None,
            "measurementMethod": None,
            "provenance": "src_a",
        }
        for period in periods
    ]


def test_dimensions_are_stat_var_roles_then_facets_in_column_order():
    doc = build_frame_doc(role_keys=("recipient",), rows=_rows(periods=["2020"]))
    view = compute_structural_view(frame=doc)
    assert [d.name for d in view.dimensions] == [
        "stat_var",
        "recipient",
        "unit",
        "measurementMethod",
        "provenance",
    ]


def test_single_value_dimension_is_not_plural():
    doc = build_frame_doc(role_keys=("recipient",), rows=_rows(periods=["2020", "2021"]))
    view = compute_structural_view(frame=doc)
    by_name = {d.name: d for d in view.dimensions}
    assert by_name["recipient"].plural is False
    assert by_name["recipient"].total_count == 1
    assert by_name["recipient"].values == ("country/ETH",)


def test_all_null_facet_has_zero_total_count_and_is_not_plural():
    doc = build_frame_doc(role_keys=("recipient",), rows=_rows(periods=["2020"]))
    view = compute_structural_view(frame=doc)
    by_name = {d.name: d for d in view.dimensions}
    assert by_name["unit"].total_count == 0
    assert by_name["unit"].values == ()
    assert by_name["unit"].plural is False


def test_a_null_beside_one_value_makes_the_dimension_plural_but_is_not_listed():
    rows = _rows(periods=["2020"]) + [
        {**row, "unit": "USDollar"} for row in _rows(periods=["2020"], recipient="country/GHA")
    ]
    doc = build_frame_doc(role_keys=("recipient",), rows=rows)
    view = compute_structural_view(frame=doc)
    unit = next(d for d in view.dimensions if d.name == "unit")
    assert unit.plural is True
    assert unit.values == ("USDollar",)
    assert unit.total_count == 1
    assert unit.truncated is False


def test_multi_value_dimension_is_plural():
    rows = _rows(periods=["2020"], recipient="country/ETH") + _rows(
        periods=["2020"], recipient="country/GHA"
    )
    doc = build_frame_doc(role_keys=("recipient",), rows=rows)
    view = compute_structural_view(frame=doc)
    by_name = {d.name: d for d in view.dimensions}
    assert by_name["recipient"].plural is True
    assert by_name["recipient"].total_count == 2
    assert set(by_name["recipient"].values) == {"country/ETH", "country/GHA"}


def test_truncation_keeps_true_count_and_sets_truncated_flag():
    rows = [row for i in range(60) for row in _rows(periods=["2020"], recipient=f"country/{i:03d}")]
    doc = build_frame_doc(role_keys=("recipient",), rows=rows)
    view = compute_structural_view(frame=doc)
    by_name = {d.name: d for d in view.dimensions}
    recipient = by_name["recipient"]
    assert recipient.total_count == 60
    assert len(recipient.values) == 50  # PER_DIMENSION_LISTING_CAP
    assert recipient.truncated is True


@pytest.mark.parametrize(
    ("period", "expected_grain"),
    [
        ("2020", "year"),
        ("2020-05", "month"),
        ("2020-05-17", "day"),
        ("2018/2022", "year"),
        ("2018-01/2022-12", "month"),
    ],
)
def test_period_grain_year_month_day(period, expected_grain):
    doc = build_frame_doc(role_keys=("recipient",), rows=_rows(periods=[period]))
    view = compute_structural_view(frame=doc)
    assert view.period_grain == expected_grain
    assert view.coverage == (period, period)


@pytest.mark.parametrize(
    "periods", [["2018/2022", "2022"], ["2018/2022", "2019/2022"], ["2017", "2018/2022"]]
)
def test_a_span_beside_any_other_period_refuses_binding_invalid(periods):
    doc = build_frame_doc(role_keys=("recipient",), rows=_rows(periods=periods))
    with pytest.raises(BindingInvalid) as excinfo:
        compute_structural_view(frame=doc)
    assert excinfo.value.pointer == "/frame_ref"


def test_coverage_and_series_count():
    doc = build_frame_doc(role_keys=("recipient",), rows=_rows(periods=["2015", "2018", "2024"]))
    view = compute_structural_view(frame=doc)
    assert view.coverage == ("2015", "2024")
    assert view.series_count == 1  # one coordinate combination, three periods


def test_series_count_counts_distinct_coordinates_not_rows():
    rows = _rows(periods=["2020", "2021"], recipient="country/ETH") + _rows(
        periods=["2020"], recipient="country/GHA"
    )
    doc = build_frame_doc(role_keys=("recipient",), rows=rows)
    view = compute_structural_view(frame=doc)
    assert view.series_count == 2


def test_empty_frame_has_no_coverage_and_default_grain():
    doc = build_frame_doc(role_keys=("recipient",), rows=[])
    view = compute_structural_view(frame=doc)
    assert view.coverage is None
    assert view.series_count == 0
    assert view.period_grain == "year"


def test_dimension_columns_excludes_time_period_value_status():
    columns = (
        "stat_var",
        "recipient",
        "time_period",
        "value",
        "unit",
        "measurementMethod",
        "provenance",
        "status",
    )
    assert dimension_columns(columns=columns) == (
        "stat_var",
        "recipient",
        "unit",
        "measurementMethod",
        "provenance",
    )


# Converted-unit dictionaries match the JSON shape loaded from a frame1 document.
# Frame.build stores UnitRequest objects, so these fixtures bypass build_frame_doc.
# The dictionaries also guard against unhashable unit cells in compute_structural_view.
_UNIT_DOC_COLUMNS = ("stat_var", "recipient", "time_period", "unit", "value")


def _unit_doc(rows):
    return {"columns": list(_UNIT_DOC_COLUMNS), "rows": [list(row) for row in rows]}


def test_converted_unit_dimension_renders_as_currency_prices_base_year():
    doc = _unit_doc(
        [
            [
                "dcid:X",
                "country/KEN",
                "2022",
                {"currency": "USD", "prices": "constant", "base_year": 2015},
                1.0,
            ],
            [
                "dcid:X",
                "country/KEN",
                "2023",
                {"currency": "USD", "prices": "constant", "base_year": 2015},
                2.0,
            ],
        ]
    )
    view = compute_structural_view(frame=doc)
    unit = next(d for d in view.dimensions if d.name == "unit")
    assert unit.values == ("USD/constant/2015",)
    assert unit.plural is False
    assert unit.total_count == 1


def test_unit_dimension_mixes_none_bare_string_and_converted_unit():
    doc = _unit_doc(
        [
            ["dcid:X", "country/KEN", "2020", None, 1.0],
            ["dcid:X", "country/KEN", "2021", "USDollar", 2.0],
            ["dcid:X", "country/KEN", "2022", {"currency": "USD", "prices": "current"}, 3.0],
        ]
    )
    view = compute_structural_view(frame=doc)
    unit = next(d for d in view.dimensions if d.name == "unit")
    # `None` is not listed; bare and rendered unit strings share one sorted listing.
    assert unit.total_count == 2
    assert unit.values == ("USD/current", "USDollar")
    assert unit.plural is True


def test_series_count_counts_two_rows_differing_only_by_converted_unit_as_two_series():
    doc = _unit_doc(
        [
            [
                "dcid:X",
                "country/KEN",
                "2022",
                {"currency": "USD", "prices": "constant", "base_year": 2015},
                1.0,
            ],
            [
                "dcid:X",
                "country/KEN",
                "2022",
                {"currency": "EUR", "prices": "constant", "base_year": 2015},
                2.0,
            ],
        ]
    )
    view = compute_structural_view(frame=doc)
    assert view.series_count == 2


def test_converted_unit_dimension_ordering_is_total_and_stable():
    doc = _unit_doc(
        [
            ["dcid:X", "country/KEN", "2020", {"currency": "USD", "prices": "current"}, 1.0],
            [
                "dcid:X",
                "country/KEN",
                "2021",
                {"currency": "EUR", "prices": "constant", "base_year": 2010},
                2.0,
            ],
            ["dcid:X", "country/KEN", "2022", "USDollar", 3.0],
            ["dcid:X", "country/KEN", "2023", None, 4.0],
        ]
    )
    view = compute_structural_view(frame=doc)
    unit = next(d for d in view.dimensions if d.name == "unit")
    assert unit.values == ("EUR/constant/2010", "USD/current", "USDollar")
    # Recomputing from the same document preserves the ordering across set iteration.
    view_again = compute_structural_view(frame=doc)
    assert view_again.dimensions == view.dimensions


_USD_PER_PERSON = {"numerator": "USDollar", "denominator": {"stat_var": "dcid:Count_Person"}}
_USD_PER_PERSON_TEXT = '{"denominator":{"stat_var":"dcid:Count_Person"},"numerator":"USDollar"}'


@pytest.mark.parametrize(
    "other",
    [
        {"numerator": "USDollar", "denominator": "dcid:Count_Person"},
        {"numerator": {"stat_var": "dcid:Count_Person"}, "denominator": "USDollar"},
        {
            "numerator": {"currency": "USD", "prices": "current"},
            "denominator": {"stat_var": "dcid:Count_Person"},
        },
        "USDollar per Count_Person",
        "USDollar",
        None,
    ],
    ids=["ref_denominator", "sides_swapped", "converted_numerator", "text_ref", "ref", "null"],
)
def test_a_composed_unit_keys_apart_from_every_other_cell(other):
    assert hashable_cell(_USD_PER_PERSON) == _USD_PER_PERSON_TEXT
    assert hashable_cell(other) != _USD_PER_PERSON_TEXT


@pytest.mark.parametrize(
    ("structured", "ref"),
    [
        (_USD_PER_PERSON, _USD_PER_PERSON_TEXT),
        ({"currency": "USD", "prices": "current"}, "USD/current"),
    ],
    ids=["composed", "converted"],
)
def test_a_unit_ref_spelling_a_structured_unit_s_key_refuses_binding_invalid(structured, ref):
    """The two cells would share one grouping key, so the frame refuses in place of grouping
    one cell's rows under the other's unit."""
    doc = _unit_doc(
        [
            ["dcid:X", "country/KEN", "2020", structured, 1.0],
            ["dcid:X", "country/KEN", "2020", ref, 99.0],
        ]
    )
    with pytest.raises(BindingInvalid) as excinfo:
        compute_structural_view(frame=doc)
    assert excinfo.value.pointer == "/frame_ref"


def test_unit_dimension_mixes_all_four_unit_forms():
    converted_per_person = {
        "numerator": {"currency": "EUR", "prices": "constant", "base_year": 2020},
        "denominator": {"stat_var": "dcid:Count_Person"},
    }
    doc = _unit_doc(
        [
            ["dcid:X", "country/KEN", "2020", None, 1.0],
            ["dcid:X", "country/KEN", "2020", "USDollar", 2.0],
            ["dcid:X", "country/KEN", "2020", {"currency": "USD", "prices": "current"}, 3.0],
            ["dcid:X", "country/KEN", "2020", _USD_PER_PERSON, 4.0],
            ["dcid:X", "country/KEN", "2021", dict(reversed(_USD_PER_PERSON.items())), 5.0],
            ["dcid:X", "country/KEN", "2020", converted_per_person, 6.0],
        ]
    )
    view = compute_structural_view(frame=doc)
    unit = next(d for d in view.dimensions if d.name == "unit")
    assert unit.values == (
        "USD/current",
        "USDollar",
        _USD_PER_PERSON_TEXT,
        '{"denominator":{"stat_var":"dcid:Count_Person"},'
        '"numerator":{"base_year":2020,"currency":"EUR","prices":"constant"}}',
    )
    assert unit.total_count == 4
    assert unit.plural is True
    assert view.series_count == 5
