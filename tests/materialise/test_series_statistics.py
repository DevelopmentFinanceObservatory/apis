"""Tests series statistics on frame-level fixtures, including sparse and mixed facets.

Frame-level fixtures cover shapes that facet selection cannot produce in the materialisation path.
"""

from __future__ import annotations

import json
import math
from typing import get_args

import pytest

from materialise.frame_build import build_frame
from materialise.statistics import (
    SERIES_STATISTICS_SCHEMA,
    EntityFigure,
    RoleFigure,
    SeriesFigures,
    SeriesStatisticsSink,
    SourceUnitFigure,
    StatusClass,
    StatusFigures,
    compute_series_statistics_table,
    count_statuses,
    method_status_classes,
)

_STAT_VAR = "dcid:MalariaGrantDisbursements"


def _facet_json(mapping: dict[str, object]) -> str:
    return json.dumps(mapping, sort_keys=True, separators=(",", ":"))


def _rows(frame) -> list[dict[str, object]]:
    table = compute_series_statistics_table(frame)
    assert table.schema.equals(SERIES_STATISTICS_SCHEMA)
    return table.to_pylist()


def test_multi_series_frame_computes_each_series_independently():
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2015",
                "value": 100.0,
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2018",
                "value": 150.0,
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2020",
                "value": 80.0,
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/KEN",
                "time_period": "2016",
                "value": 50.0,
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/KEN",
                "time_period": "2019",
                "value": 70.0,
            },
        ],
    )

    rows = _rows(frame)
    by_recipient = {row["entities"][0]["ref"]: row for row in rows}
    assert set(by_recipient) == {"dcid:country/TGO", "dcid:country/KEN"}

    tgo = by_recipient["dcid:country/TGO"]
    assert tgo["observations"] == 3
    assert (tgo["first_value"], tgo["first_period"]) == (100.0, "2015")
    assert (tgo["latest_value"], tgo["latest_period"]) == (80.0, "2020")
    assert (tgo["minimum_value"], tgo["minimum_period"]) == (80.0, "2020")
    assert (tgo["maximum_value"], tgo["maximum_period"]) == (150.0, "2018")
    assert tgo["change_absolute"] == -20.0
    assert tgo["change_percent"] == -20.0

    ken = by_recipient["dcid:country/KEN"]
    assert ken["observations"] == 2
    assert (ken["first_value"], ken["first_period"]) == (50.0, "2016")
    assert (ken["latest_value"], ken["latest_period"]) == (70.0, "2019")
    assert ken["change_absolute"] == 20.0
    assert ken["change_percent"] == 40.0


def test_interleaved_bound_facet_frame_groups_by_full_grain_tuple():
    # Two facet combos over the SAME stat_var+recipient. `row_sort_key`
    # orders rows by time_period BEFORE the facet columns, so the two
    # combos' rows interleave in `frame.rows` (EUR/2015, USD/2015,
    # EUR/2016, USD/2016) rather than sitting contiguously -- grouping must
    # still recover each series correctly.
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2015",
                "value": 10.0,
                "unit": "EUR",
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2015",
                "value": 20.0,
                "unit": "USD",
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2016",
                "value": 15.0,
                "unit": "EUR",
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2016",
                "value": 25.0,
                "unit": "USD",
            },
        ],
    )
    # Pin the interleaving this test relies on -- if `Frame.build`'s own
    # collation ever changed, this fixture would stop exercising the case.
    assert [row[frame.columns.index("unit")] for row in frame.rows] == [
        "EUR",
        "USD",
        "EUR",
        "USD",
    ]

    rows = _rows(frame)
    by_unit = {json.loads(row["facet"])["unit"]: row for row in rows}
    assert set(by_unit) == {"EUR", "USD"}

    eur = by_unit["EUR"]
    assert eur["observations"] == 2
    assert (eur["first_value"], eur["first_period"]) == (10.0, "2015")
    assert (eur["latest_value"], eur["latest_period"]) == (15.0, "2016")

    usd = by_unit["USD"]
    assert usd["observations"] == 2
    assert (usd["first_value"], usd["first_period"]) == (20.0, "2015")
    assert (usd["latest_value"], usd["latest_period"]) == (25.0, "2016")


def test_single_observation_series_has_null_change():
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2015",
                "value": 42.0,
            }
        ],
    )

    (row,) = _rows(frame)
    assert row["observations"] == 1
    assert (row["first_value"], row["first_period"]) == (row["latest_value"], row["latest_period"])
    assert row["change_absolute"] is None
    assert row["change_percent"] is None


def test_zero_negative_and_sign_crossing_baselines_null_percent_but_keep_absolute():
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/ZERO",
                "time_period": "2015",
                "value": 0.0,
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/ZERO",
                "time_period": "2020",
                "value": 10.0,
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/NEG",
                "time_period": "2015",
                "value": -5.0,
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/NEG",
                "time_period": "2020",
                "value": -2.0,
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/NEGZERO",
                "time_period": "2015",
                "value": -0.0,
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/NEGZERO",
                "time_period": "2020",
                "value": 5.0,
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/CROSS",
                "time_period": "2015",
                "value": -5.0,
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/CROSS",
                "time_period": "2020",
                "value": 5.0,
            },
        ],
    )

    by_recipient = {row["entities"][0]["ref"]: row for row in _rows(frame)}

    zero = by_recipient["dcid:country/ZERO"]
    assert zero["change_absolute"] == 10.0
    assert zero["change_percent"] is None

    negative = by_recipient["dcid:country/NEG"]
    assert negative["change_absolute"] == 3.0
    assert negative["change_percent"] is None

    negative_zero = by_recipient["dcid:country/NEGZERO"]
    assert negative_zero["change_absolute"] == 5.0
    assert negative_zero["change_percent"] is None

    sign_crossing = by_recipient["dcid:country/CROSS"]
    assert sign_crossing["change_absolute"] == 10.0
    assert sign_crossing["change_percent"] is None


def test_minimum_and_maximum_ties_break_to_the_earliest_period():
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2015",
                "value": 10.0,
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2016",
                "value": 10.0,
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2018",
                "value": 5.0,
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2020",
                "value": 5.0,
            },
        ],
    )

    (row,) = _rows(frame)
    assert (row["maximum_value"], row["maximum_period"]) == (10.0, "2015")
    assert (row["minimum_value"], row["minimum_period"]) == (5.0, "2018")


def test_change_absolute_is_decimal_clean_not_a_binary_float_artifact():
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2015",
                "value": 12.4,
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2024",
                "value": 9.8,
            },
        ],
    )

    (row,) = _rows(frame)
    # A plain binary-float subtraction gives -2.5999999999999996; the
    # decimal-clean computation must land on exactly -2.6.
    assert row["change_absolute"] == -2.6
    assert 9.8 - 12.4 != -2.6  # pins the artifact this computation avoids


def test_percent_is_exactly_recomputable_from_absolute_and_published_first():
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2015",
                "value": 12.4,
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2024",
                "value": 9.8,
            },
        ],
    )

    (row,) = _rows(frame)
    assert row["change_percent"] == row["change_absolute"] / abs(row["first_value"]) * 100


def test_mixed_none_and_string_facet_cells_do_not_raise_and_facet_stays_sparse():
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2015",
                "value": 10.0,
                "unit": "USD",
                "measurementMethod": None,
                "provenance": "OECD_DAC",
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2016",
                "value": 20.0,
                "unit": None,
                "measurementMethod": "Method1",
                "provenance": None,
            },
        ],
    )

    facets = {row["first_period"]: json.loads(row["facet"]) for row in _rows(frame)}
    assert facets["2015"] == {"unit": "USD", "provenance": "OECD_DAC"}
    assert facets["2016"] == {"measurementMethod": "Method1"}


def test_unit_request_cell_is_projected_via_as_json():
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2015",
                "value": 10.0,
                "unit": {"currency": "USD", "prices": "current"},
            }
        ],
    )

    (row,) = _rows(frame)
    assert json.loads(row["facet"])["unit"] == {"currency": "USD", "prices": "current"}


def test_series_that_differ_only_in_a_composed_unit_stay_apart():
    per_person = {"numerator": "USDollar", "denominator": {"stat_var": "dcid:Count_Person"}}
    per_woman = {"numerator": "USDollar", "denominator": {"stat_var": "dcid:Count_Person_Female"}}
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2015",
                "value": value,
                "unit": unit,
            }
            for unit, value in ((per_person, 10.0), (per_woman, 20.0))
        ],
    )

    rows = _rows(frame)

    assert [json.loads(row["facet"])["unit"] for row in rows] == [per_person, per_woman]
    assert [row["observations"] for row in rows] == [1, 1]


def test_a_frame_mixing_all_four_unit_forms_keeps_one_series_per_unit():
    units = [
        None,
        "USDollar",
        {"currency": "USD", "prices": "current"},
        {"numerator": "USDollar", "denominator": {"stat_var": "dcid:Count_Person"}},
    ]
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": period,
                "value": 1.0,
                "unit": unit,
            }
            for unit in units
            for period in ("2015", "2016")
        ],
    )

    rows = _rows(frame)

    assert sorted(
        json.dumps(json.loads(row["facet"]).get("unit"), sort_keys=True) for row in rows
    ) == sorted(json.dumps(unit, sort_keys=True) for unit in units)
    assert [row["observations"] for row in rows] == [2] * 4


def test_facet_json_has_sorted_keys_and_no_whitespace():
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2015",
                "value": 10.0,
                "unit": "USD",
                "measurementMethod": "Method1",
                "provenance": "OECD_DAC",
            }
        ],
    )

    (row,) = _rows(frame)
    assert row["facet"] == _facet_json(
        {"unit": "USD", "measurementMethod": "Method1", "provenance": "OECD_DAC"}
    )
    assert " " not in row["facet"]


def test_statistics_are_independent_of_input_row_order():
    rows = [
        {
            "stat_var": _STAT_VAR,
            "recipient": "dcid:country/TGO",
            "time_period": "2015",
            "value": 10.0,
        },
        {
            "stat_var": _STAT_VAR,
            "recipient": "dcid:country/TGO",
            "time_period": "2018",
            "value": 15.0,
        },
        {
            "stat_var": _STAT_VAR,
            "recipient": "dcid:country/KEN",
            "time_period": "2016",
            "value": 50.0,
        },
    ]

    forward = _rows(build_frame(role_keys=("recipient",), rows=rows))
    shuffled = _rows(build_frame(role_keys=("recipient",), rows=list(reversed(rows))))
    assert forward == shuffled


def test_negative_zero_value_emits_positive_zero():
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2015",
                "value": -0.0,
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2020",
                "value": 5.0,
            },
        ],
    )

    (row,) = _rows(frame)
    # `-0.0 == 0.0` is `True` in Python, so the sign has to be checked
    # explicitly rather than with a plain equality assertion.
    assert math.copysign(1.0, row["first_value"]) == 1.0
    assert math.copysign(1.0, row["minimum_value"]) == 1.0


def test_mixed_granularity_within_one_series_orders_lexicographically():
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2024-03",
                "value": 200.0,
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2024",
                "value": 100.0,
            },
        ],
    )

    (row,) = _rows(frame)
    # "2024" < "2024-03" lexicographically (a prefix sorts first), so
    # despite the input order above, "2024" is first and "2024-03" is
    # latest.
    assert (row["first_value"], row["first_period"]) == (100.0, "2024")
    assert (row["latest_value"], row["latest_period"]) == (200.0, "2024-03")


def test_zero_row_frame_returns_an_empty_table_with_the_schema_intact():
    frame = build_frame(role_keys=("recipient",), rows=())
    table = compute_series_statistics_table(frame)
    assert table.schema.equals(SERIES_STATISTICS_SCHEMA)
    assert table.num_rows == 0


def test_absolute_overflow_nulls_the_whole_change():
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2015",
                "value": -1e308,
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2020",
                "value": 1e308,
            },
        ],
    )

    (row,) = _rows(frame)
    assert row["observations"] == 2
    assert (row["first_value"], row["first_period"]) == (-1e308, "2015")
    assert (row["latest_value"], row["latest_period"]) == (1e308, "2020")
    assert (row["minimum_value"], row["minimum_period"]) == (-1e308, "2015")
    assert (row["maximum_value"], row["maximum_period"]) == (1e308, "2020")
    assert row["change_absolute"] is None
    assert row["change_percent"] is None


def test_percent_overflow_nulls_percent_but_keeps_finite_absolute():
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2015",
                "value": 1e-308,
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2020",
                "value": 1e308,
            },
        ],
    )

    (row,) = _rows(frame)
    assert row["change_absolute"] == 1e308
    assert row["change_percent"] is None


def test_overflowed_change_rows_still_serialise_only_finite_floats():
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2015",
                "value": -1e308,
            },
            {
                "stat_var": _STAT_VAR,
                "recipient": "dcid:country/TGO",
                "time_period": "2020",
                "value": 1e308,
            },
        ],
    )

    (row,) = _rows(frame)
    # `allow_nan=False` raises on the first `inf`/`nan` it hits, so a clean
    # round trip over every non-null numeric field is proof each one is
    # finite.
    numeric_fields = {
        k: v for k, v in row.items() if k not in {"entities", "facet"} and v is not None
    }
    json.loads(json.dumps(numeric_fields, allow_nan=False))


def test_multi_role_frame_orders_entities_in_frame_column_order():
    frame = build_frame(
        role_keys=("financingProvider", "financingRecipient"),
        rows=[
            {
                "stat_var": _STAT_VAR,
                "financingProvider": "dcid:country/USA",
                "financingRecipient": "dcid:country/TGO",
                "time_period": "2015",
                "value": 10.0,
            },
            {
                "stat_var": _STAT_VAR,
                "financingProvider": "dcid:country/USA",
                "financingRecipient": "dcid:country/TGO",
                "time_period": "2016",
                "value": 20.0,
            },
            {
                "stat_var": _STAT_VAR,
                "financingProvider": "dcid:country/FRA",
                "financingRecipient": "dcid:country/TGO",
                "time_period": "2015",
                "value": 5.0,
            },
        ],
    )

    rows = _rows(frame)
    usa = next(
        row
        for row in rows
        if {"role": "financingProvider", "ref": "dcid:country/USA"} in row["entities"]
    )
    assert usa["observations"] == 2
    assert usa["entities"] == [
        {"role": "financingProvider", "ref": "dcid:country/USA"},
        {"role": "financingRecipient", "ref": "dcid:country/TGO"},
    ]
    fra = next(
        row
        for row in rows
        if {"role": "financingProvider", "ref": "dcid:country/FRA"} in row["entities"]
    )
    assert fra["observations"] == 1


def _row(recipient: str, period: str, **facet: object) -> dict[str, object]:
    return {
        "stat_var": _STAT_VAR,
        "recipient": recipient,
        "time_period": period,
        "value": 1.0,
        **facet,
    }


def _figures(*chunks: list[dict[str, object]]) -> SeriesFigures:
    sink = SeriesStatisticsSink(continuing=())
    for rows in chunks:
        sink.write(
            compute_series_statistics_table(build_frame(role_keys=("recipient",), rows=rows))
        )
    return sink.finish().figures


def test_every_packaged_method_names_a_status_class():
    classes = method_status_classes()

    assert classes
    assert set(classes.values()) <= set(get_args(StatusClass))


@pytest.mark.parametrize(
    ("methods", "expected"),
    [
        pytest.param(("Estimated", "Modelled", "Estimated"), StatusFigures(3, 0), id="all-classed"),
        pytest.param(("Estimated", None, "Modelled"), StatusFigures(None, None), id="one-missing"),
        pytest.param(
            ("Estimated", "WikidataPopulation"), StatusFigures(None, None), id="one-unclassed"
        ),
    ],
)
def test_status_counts_are_known_only_when_every_method_is_classed(methods, expected):
    rows = [
        _row(f"dcid:country/C{index}", "2020", measurementMethod=method)
        for index, method in enumerate(methods)
    ]

    assert _figures(rows).status == expected


_LISTED = "provenance/Listed"
_UNLISTED = "provenance/Unlisted"


@pytest.mark.parametrize(
    ("values_by_facet", "expected"),
    [
        pytest.param(
            {("Draft", _UNLISTED): 2, ("Survey", _UNLISTED): 5, ("Model", _UNLISTED): 1},
            StatusFigures(1, 2),
            id="classed",
        ),
        pytest.param(
            {("Survey", _UNLISTED): 5}, StatusFigures(0, 0), id="reported-only-is-a-known-zero"
        ),
        pytest.param({}, StatusFigures(0, 0), id="no-values"),
        pytest.param(
            {("Survey", _UNLISTED): 5, (None, _UNLISTED): 1},
            StatusFigures(None, None),
            id="missing-under-an-unlisted-provenance",
        ),
        pytest.param(
            {("Model", _LISTED): 3, (None, _LISTED): 4},
            StatusFigures(3, 0),
            id="missing-under-a-listed-provenance-is-reported",
        ),
        pytest.param(
            {(None, _LISTED): 4, (None, _UNLISTED): 1},
            StatusFigures(None, None),
            id="one-unlisted-missing-makes-both-unknown",
        ),
        pytest.param(
            {("Survey", _UNLISTED): 5, ("Other", _UNLISTED): 1},
            StatusFigures(None, None),
            id="unclassed",
        ),
    ],
)
def test_count_statuses_sums_values_per_class(values_by_facet, expected):
    classes: dict[str, StatusClass] = {
        "Draft": "provisional",
        "Survey": "reported",
        "Model": "estimated",
    }

    assert (
        count_statuses(values_by_facet, classes=classes, reported=frozenset({_LISTED})) == expected
    )


@pytest.mark.parametrize(
    ("chunks", "span", "grain", "reaching"),
    [
        pytest.param(
            [
                [
                    _row("dcid:country/TGO", "2015"),
                    _row("dcid:country/TGO", "2020"),
                    _row("dcid:country/KEN", "2018"),
                ],
                [_row("dcid:country/GHA", "2016"), _row("dcid:country/GHA", "2020")],
            ],
            ("2015", "2020"),
            "year",
            2,
            id="two-of-three-reach-across-chunks",
        ),
        pytest.param(
            [[_row("dcid:country/TGO", "2019/2021"), _row("dcid:country/KEN", "2021")]],
            ("2019", "2021"),
            "year",
            2,
            id="an-interval-reaches-by-its-end",
        ),
        pytest.param(
            [[_row("dcid:country/TGO", "2020"), _row("dcid:country/KEN", "2020-06")]],
            None,
            None,
            None,
            id="mixed-grains-do-not-order",
        ),
        pytest.param([[]], None, None, None, id="no-series"),
    ],
)
def test_latest_period_span_and_grain(chunks, span, grain, reaching):
    figures = _figures(*chunks)

    assert (figures.span, figures.grain) == (span, grain)
    assert figures.latest_period == (span[1] if span else None)
    assert figures.series_reaching_latest == reaching


def test_series_group_by_source_and_unit():
    converted = {"currency": "USD", "prices": "constant", "base_year": 2022}
    rows = [
        _row("dcid:country/TGO", "2020", provenance="provenance/CRS", unit="USDollar"),
        _row("dcid:country/KEN", "2020", provenance="provenance/CRS", unit="USDollar"),
        _row("dcid:country/TGO", "2020", provenance="provenance/CRS", unit=converted),
        _row("dcid:country/TGO", "2020", provenance="provenance/IDS", unit="USDollar"),
        _row("dcid:country/GHA", "2020", provenance="provenance/IDS"),
    ]

    assert sorted(_figures(rows[:3], rows[3:]).by_source_unit, key=repr) == sorted(
        [
            SourceUnitFigure(source="provenance/CRS", unit="USDollar", series=2),
            SourceUnitFigure(source="provenance/CRS", unit=converted, series=1),
            SourceUnitFigure(source="provenance/IDS", unit="USDollar", series=1),
            SourceUnitFigure(source="provenance/IDS", unit=None, series=1),
        ],
        key=repr,
    )


def test_entities_per_role_count_distinct_refs_across_chunks():
    figures = _figures(
        [_row("dcid:country/TGO", "2020"), _row("dcid:country/KEN", "2020")],
        [_row("dcid:country/TGO", "2020", measurementMethod="Estimated")],
    )

    assert figures.series == 3
    assert [(figure.role, figure.entities) for figure in figures.entities_by_role] == [
        ("recipient", 2)
    ]


def test_each_roles_entities_count_the_distinct_variables_holding_them_across_chunks():
    roles = ("financingProvider", "financingRecipient")

    def row(stat_var: str, provider: str, recipient: str, **facet: object) -> dict[str, object]:
        return {
            "stat_var": stat_var,
            "financingProvider": provider,
            "financingRecipient": recipient,
            "time_period": "2020",
            "value": 1.0,
            **facet,
        }

    sink = SeriesStatisticsSink(continuing=())
    for rows in (
        [
            row("dcid:A", "dcid:P1", "dcid:country/TGO"),
            # A second series of the same variable and entities counts the variable once.
            row("dcid:A", "dcid:P1", "dcid:country/TGO", measurementMethod="Estimated"),
            row("dcid:A", "dcid:P2", "dcid:country/KEN"),
        ],
        [row("dcid:B", "dcid:P1", "dcid:country/KEN")],
    ):
        sink.write(compute_series_statistics_table(build_frame(role_keys=roles, rows=rows)))
    figures = sink.finish().figures

    assert figures.series == 4
    assert figures.entities_by_role == (
        RoleFigure(
            role="financingProvider",
            values=(
                EntityFigure(ref="dcid:P1", variables=2),
                EntityFigure(ref="dcid:P2", variables=1),
            ),
        ),
        RoleFigure(
            role="financingRecipient",
            values=(
                EntityFigure(ref="dcid:country/TGO", variables=1),
                EntityFigure(ref="dcid:country/KEN", variables=2),
            ),
        ),
    )
