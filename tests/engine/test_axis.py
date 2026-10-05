"""Test axis overrides and date/place classification rules."""

import pytest

from search.engine.axis import AXIS_OVERRIDES, classify_axis


def test_comparison_region_is_where() -> None:
    assert AXIS_OVERRIDES["comparisonRegion"] == "where"


def test_place_named_props_are_how() -> None:
    for prop in (
        "placeOfBirth",
        "placeOfResidenceClassification",
        "placeOfWork",
        "placeCategory",
        "locationType",
        "jurisdiction",
        "computerUsageLocation",
        "internetUsageLocation",
    ):
        assert AXIS_OVERRIDES[prop] == "how", f"{prop!r} should be 'how'"


def test_time_named_props_are_how() -> None:
    for prop in (
        "dateBuilt",
        "periodOfMilitaryService",
        "instrumentTerm",
        "maturity",
        "commuteTime",
        "accumulationPeriod",
        "extremesOverTime",
    ):
        assert AXIS_OVERRIDES[prop] == "how", f"{prop!r} should be 'how'"


def test_classify_place_of_birth_override() -> None:
    # placeOfBirth is a how axis, even for place values.
    result = classify_axis("placeOfBirth", ["country/USA", "country/FRA"])
    assert result == "how"


@pytest.mark.parametrize(
    "date_vals",
    [
        ["2020", "2021"],
        ["2020-01", "2020-06"],
        ["2020-01-15"],
    ],
)
def test_classify_date_iso_formats(date_vals: list[str]) -> None:
    assert classify_axis("someDateProp", date_vals) == "when"


def test_classify_date_mixed_does_not_trigger_on_non_date() -> None:
    # Non-date values use the how default.
    assert classify_axis("someHowProp", ["Rural", "Urban"]) == "how"


def test_classify_all_place_values_is_where() -> None:
    values = [
        "country/USA",
        "country/ETH",
        "country/KEN",
        "country/FRA",
        "country/DEU",
        "country/GBR",
        "country/IND",
        "country/CHN",
        "country/BRA",
        "country/ZAF",
    ]
    assert classify_axis("someGeoProperty", values) == "where"


def test_classify_exactly_90pct_place_is_where() -> None:
    # Nine of ten place values meet the 0.9 threshold.
    values = ["country/USA"] * 9 + ["SomeEnum"]
    assert classify_axis("nearlyGeo", values) == "where"


def test_classify_below_90pct_place_is_how() -> None:
    # Eight of ten falls below the 0.9 threshold.
    values = ["country/USA"] * 8 + ["SomeEnum", "AnotherEnum"]
    assert classify_axis("mixedGeo", values) == "how"


def test_classify_no_place_values_is_how() -> None:
    values = ["ODAGrants", "ODALoans", "PrivateFlows"]
    assert classify_axis("myProperty", values) == "how"


def test_classify_empty_values_is_how() -> None:
    assert classify_axis("anyProperty", []) == "how"


def test_classify_purpose_values_is_how() -> None:
    # org/ values use the how default.
    values = ["org/Health", "org/BasicHealth", "org/Agriculture"]
    assert classify_axis("grantPurpose", values) == "how"
