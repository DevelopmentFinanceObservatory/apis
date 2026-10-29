"""`run_paired_compare` and `run_paired_sum_of`: members paired by `on`, the
dropped-member accounting, the output keyed as `data` names it, and each
refusal."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import pytest
from dfo_frame_grammar import Frame

from dfo_analysis_engine.engine.aggregate import ComposedUnitSumError
from dfo_analysis_engine.engine.compare import (
    CompareAmbiguousSliceError,
    CompareResult,
    CompareUnitMismatchError,
    ComparisonPeriodsDisjointError,
)
from dfo_analysis_engine.engine.paired import (
    PairedNoKeyMatchedError,
    PairedResult,
    PairedUnknownColumnError,
    run_paired_compare,
    run_paired_sum_of,
)
from dfo_analysis_engine.engine.sum_of import SumOfResult

_OWN_REF = "dcid:ops_" + "e" * 32
_USD = "dcid:USDollar"
_ON = {"destination": "observationAbout"}


def _crs(*cells: tuple[str, str, float], unit: object = _USD) -> Frame:
    """ODA by recipient, the `data` side: `(destination, year, value)` cells."""
    return Frame.build(
        role_keys=["destination"],
        rows=[
            {
                "stat_var": "dcid:Amount_FinancialFlow_ODA_Disbursement",
                "destination": place,
                "time_period": year,
                "value": value,
                "unit": unit,
            }
            for place, year, value in cells
        ],
    )


def _wdi(*cells: tuple[str, str, float], unit: object = _USD) -> Frame:
    """Net ODA by country, the `other` side: `(observationAbout, year, value)` cells."""
    return Frame.build(
        role_keys=["observationAbout"],
        rows=[
            {
                "stat_var": "dcid:Amount_FinancialFlow_ODA_NetDisbursement",
                "observationAbout": place,
                "time_period": year,
                "value": value,
                "unit": unit,
            }
            for place, year, value in cells
        ],
    )


def _sum(data: Frame, other: Frame, on: dict[str, str] = _ON) -> PairedResult:
    return run_paired_sum_of(data, other, stat_var=_OWN_REF, on=on, left_entity={}, right_entity={})


def _compare(data: Frame, other: Frame, on: dict[str, str] = _ON) -> PairedResult:
    """`data` minus `other`: `data` fills compare's right side."""
    return run_paired_compare(
        other, data, stat_var=_OWN_REF, on=on, left_entity={}, right_entity={}
    )


def _rows(result: PairedResult) -> list[tuple[str, str, float]]:
    """Each output row's member, period and value; every row carries `result.unit`."""
    assert result.unit == _USD
    columns = result.frame.table.to_pydict()
    return sorted(
        zip(columns["destination"], columns["time_period"], columns["value"], strict=True)
    )


_DATA = _crs(("country/KEN", "2022", 300.0), ("country/KEN", "2023", 250.0))
_OTHER = _wdi(("country/KEN", "2022", 280.0), ("country/KEN", "2023", 265.5))


@pytest.mark.parametrize(
    ("run", "values"),
    [
        pytest.param(_sum, (580.0, 515.5), id="sum_of-adds"),
        pytest.param(_compare, (20.0, -15.5), id="compare-data-minus-other"),
    ],
)
def test_paired_members_give_one_row_per_member_and_period_keyed_as_data_names_it(
    run: Callable[[Frame, Frame], PairedResult], values: tuple[float, float]
) -> None:
    """The spec-kit `operation-outputs` vectors `sum-of-paired` and
    `compare-data-minus-other`."""
    result = run(_DATA, _OTHER)

    assert _rows(result) == [
        ("country/KEN", "2022", values[0]),
        ("country/KEN", "2023", values[1]),
    ]
    assert result.frame.columns[:2] == ("stat_var", "destination")
    assert set(result.frame.table.to_pydict()["stat_var"]) == {_OWN_REF}
    assert result.intersected_periods == ("2022", "2023")
    assert (result.left_dropped_key_count, result.right_dropped_key_count) == (0, 0)


@pytest.mark.parametrize(
    ("single", "per_period"),
    [(CompareResult, {"period_comparisons"}), (SumOfResult, {"period_sums", "left_out"})],
    ids=["compare", "sum_of"],
)
def test_a_paired_result_names_every_field_the_single_series_result_does(
    single: type, per_period: set[str]
) -> None:
    """A summary renders from these fields, so a paired call reads as its verb
    does but for the fields kept per period, which a paired result keeps per
    member (`cells`, `left_only_periods` and `right_only_periods`)."""
    paired = {field.name for field in dataclasses.fields(PairedResult)}

    assert {field.name for field in dataclasses.fields(single)} - per_period <= paired


def test_data_fills_compares_right_side_and_sum_ofs_left() -> None:
    assert _compare(_DATA, _OTHER).data_side == "right"
    assert _sum(_DATA, _OTHER).data_side == "left"


@pytest.mark.parametrize("run", [_sum, _compare], ids=["sum_of", "compare"])
def test_a_member_one_side_lacks_is_left_out_and_counted_on_its_side(
    run: Callable[[Frame, Frame], PairedResult],
) -> None:
    data = _crs(("country/KEN", "2022", 300.0), ("country/GHA", "2022", 90.0))
    other = _wdi(
        ("country/KEN", "2022", 280.0),
        ("country/NGA", "2022", 50.0),
        ("country/NGA", "2023", 55.0),
    )

    result = run(data, other)

    assert [row[0] for row in _rows(result)] == ["country/KEN"]
    data_side, other_side = ("left", "right") if result.data_side == "left" else ("right", "left")
    unmatched = {
        "left": result.left_unmatched_keys,
        "right": result.right_unmatched_keys,
    }
    assert unmatched[data_side] == ((("destination", "country/GHA"),),)
    assert unmatched[other_side] == ((("observationAbout", "country/NGA"),),)
    assert (result.left_dropped_key_count, result.right_dropped_key_count) == (1, 1)
    records = {
        (record.reason, record.side): (record.row_count, record.entities)
        for record in result.exclusions.records
    }
    assert records == {
        ("unmatched_key", data_side): (1, {"destination": ("country/GHA",)}),
        ("unmatched_key", other_side): (2, {"observationAbout": ("country/NGA",)}),
    }


def test_a_paired_member_sharing_no_period_is_dropped_from_both_sides() -> None:
    data = _crs(
        ("country/KEN", "2022", 300.0),
        ("country/KEN", "2023", 250.0),
        ("country/UGA", "2021", 40.0),
    )
    other = _wdi(("country/KEN", "2023", 265.5), ("country/UGA", "2023", 60.0))

    result = _sum(data, other)

    assert _rows(result) == [("country/KEN", "2023", 515.5)]
    assert result.keys_without_shared_period == ((("destination", "country/UGA"),),)
    assert (result.left_dropped_key_count, result.right_dropped_key_count) == (1, 1)
    assert result.left_only_periods == (((("destination", "country/KEN"),), ("2022",)),)
    assert result.left_coverage_fraction == 1 / 3
    assert result.right_coverage_fraction == 1 / 2
    period_unmatched = {
        record.side: record.row_count
        for record in result.exclusions.records
        if record.reason == "period_unmatched"
    }
    assert period_unmatched == {"left": 2, "right": 1}


def test_a_column_both_sides_name_alike_pairs_with_itself() -> None:
    data = _crs(("country/KEN", "2022", 300.0))
    other = _crs(("country/KEN", "2022", 20.0))

    result = _sum(data, other, on={"destination": "destination"})

    assert _rows(result) == [("country/KEN", "2022", 320.0)]


_PER_HEAD = {"numerator": _USD, "denominator": {"stat_var": "dcid:Count_Person"}}


@pytest.mark.parametrize(
    ("run", "data", "other", "on", "error", "attributes"),
    [
        pytest.param(
            _sum,
            _crs(("country/KEN", "2022", 1.0)),
            _wdi(("country/NGA", "2022", 1.0)),
            _ON,
            PairedNoKeyMatchedError,
            {
                "left_keys": ((("destination", "country/KEN"),),),
                "right_keys": ((("observationAbout", "country/NGA"),),),
                "left_key_count": 1,
                "right_key_count": 1,
            },
            id="no-member-pairs",
        ),
        pytest.param(
            _compare,
            _crs(("country/KEN", "2022", 1.0)),
            _wdi(("country/KEN", "2022", 1.0)),
            {"recipient": "observationAbout"},
            PairedUnknownColumnError,
            {"side": "right", "columns": ("recipient",)},
            id="on-names-a-column-data-lacks",
        ),
        pytest.param(
            _sum,
            _crs(("country/KEN", "2022", 1.0)),
            _wdi(("country/KEN", "2023", 1.0)),
            _ON,
            ComparisonPeriodsDisjointError,
            {"left_periods": ("2022",), "right_periods": ("2023",)},
            id="paired-member-shares-no-period",
        ),
        pytest.param(
            _compare,
            _crs(("country/KEN", "2022", 1.0)),
            _wdi(("country/KEN", "2022", 1.0), unit="dcid:Percent"),
            _ON,
            CompareUnitMismatchError,
            {"left_unit": "dcid:Percent", "right_unit": _USD},
            id="units-differ",
        ),
        pytest.param(
            _sum,
            _crs(("country/KEN", "2022", 1.0), unit=_PER_HEAD),
            _wdi(("country/KEN", "2022", 1.0), unit=_PER_HEAD),
            _ON,
            ComposedUnitSumError,
            {"side": "left"},
            id="sum-of-composed-units",
        ),
    ],
)
def test_refusals(
    run: Callable[..., PairedResult],
    data: Frame,
    other: Frame,
    on: dict[str, str],
    error: type[Exception],
    attributes: dict[str, Any],
) -> None:
    with pytest.raises(error) as raised:
        run(data, other, on)
    for name, expected in attributes.items():
        assert getattr(raised.value, name) == expected


def test_two_rows_for_one_member_and_period_refuse_as_ambiguous() -> None:
    data = Frame.build(
        role_keys=["destination", "origin"],
        rows=[
            {
                "stat_var": "dcid:X",
                "destination": "country/KEN",
                "origin": donor,
                "time_period": "2022",
                "value": 1.0,
                "unit": _USD,
            }
            for donor in ("country/DEU", "country/FRA")
        ],
    )

    with pytest.raises(CompareAmbiguousSliceError) as raised:
        _sum(data, _OTHER)

    assert (raised.value.side, raised.value.by) == ("left", ("destination",))
