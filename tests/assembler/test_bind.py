"""Pin period binding, status passthrough, and unit, source, and label resolution.

A hand-built frame exercises non-null status, which `Frame.build` currently sets to `None`.
"""

from __future__ import annotations

import datetime

import pytest

from assembler.bind import bind_all, bind_series, count_periods, frame_periods
from assembler.frame_view import compute_structural_view
from assembler.ports import FrameRef, ResolvedFrame
from assembler.resolve import resolve_selectors
from assembler.testing import FIXTURE_RETRIEVED_ON, build_resolved_frame

from ._pipeline_support import _stored_resolved_frame

_FRAME_REF = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32)


def _row(*, recipient="country/ETH", period, value=1.0, unit=None, provenance="src_a"):
    return {
        "stat_var": "dcid:X",
        "recipient": recipient,
        "time_period": period,
        "value": value,
        "unit": unit,
        "measurementMethod": None,
        "provenance": provenance,
    }


def test_missing_period_becomes_typed_missing_point():
    rf = build_resolved_frame(
        frame_ref=_FRAME_REF,
        role_keys=("recipient",),
        rows=[_row(period="2020"), _row(period="2022")],
    )
    (series,) = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]}],
        window={"start": "2020", "end": "2022"},
    )
    (bound,) = bind_all(resolved_frame=rf, series=[series], window={"start": "2020", "end": "2022"})
    points_by_period = {p["period"]: p for p in bound.points}
    assert points_by_period["2020"] == {"period": "2020", "value": 1.0}
    assert points_by_period["2021"] == {"period": "2021", "status": "missing"}
    assert points_by_period["2022"] == {"period": "2022", "value": 1.0}


def test_missing_point_is_never_zero_or_interpolated():
    rf = build_resolved_frame(
        frame_ref=_FRAME_REF,
        role_keys=("recipient",),
        rows=[_row(period="2020"), _row(period="2022")],
    )
    (series,) = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]}],
        window={"start": "2020", "end": "2022"},
    )
    (bound,) = bind_all(resolved_frame=rf, series=[series], window={"start": "2020", "end": "2022"})
    gap = next(p for p in bound.points if p["period"] == "2021")
    assert "value" not in gap
    assert gap["status"] == "missing"


def test_status_the_frame_carries_passes_through():
    # dfo_frame_grammar.Frame.build always forces status=None; a status
    # frame-grammar might carry once a future minor version populates it is
    # constructed here directly, to exercise bind.py's passthrough branch.
    doc = {
        "frame_version": "frame1",
        "columns": [
            "stat_var",
            "recipient",
            "time_period",
            "value",
            "unit",
            "measurementMethod",
            "provenance",
            "status",
        ],
        "rows": [["dcid:X", "country/ETH", "2020", 1.0, None, None, "src_a", "suppressed"]],
    }
    rf = ResolvedFrame(
        frame_ref=_FRAME_REF,
        frame=doc,
        structural_view=compute_structural_view(frame=doc),
        manifest={},
        labels={},
        retrieved_on=FIXTURE_RETRIEVED_ON,
    )
    (series,) = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]}],
        window={"start": "2020", "end": "2020"},
    )
    (bound,) = bind_all(resolved_frame=rf, series=[series], window={"start": "2020", "end": "2020"})
    assert bound.points == ({"period": "2020", "status": "suppressed"},)


def test_a_series_label_keeps_every_role_when_one_role_label_fills_the_limit():
    """Two series share a 200-character donor label and differ in recipient."""
    rf = build_resolved_frame(
        frame_ref=_FRAME_REF,
        role_keys=("donor", "recipient"),
        rows=[
            {**_row(recipient=recipient, period="2020"), "donor": "country/USA"}
            for recipient in ("country/ETH", "country/UGA")
        ],
        labels={"country/USA": "D" * 200, "country/ETH": "Ethiopia", "country/UGA": "Uganda"},
    )
    window = {"start": "2020", "end": "2020"}
    series = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "donor", "values": ["country/USA"]}]}],
        window=window,
    )
    bound = bind_all(resolved_frame=rf, series=series, window=window)
    assert [one.label for one in bound] == [
        "D" * 189 + " / Ethiopia",
        "D" * 191 + " / Uganda",
    ]


@pytest.mark.parametrize(
    ("labels", "expected_unit"),
    [
        pytest.param({}, "unit/USD", id="unlabelled-ref-stays-raw"),
        pytest.param({"unit/USD": "US Dollar"}, "US Dollar", id="labelled-ref"),
        pytest.param({"unit/USD": "U" * 201}, "U" * 200, id="long-label-is-cut"),
    ],
)
def test_unit_label_and_source_ref_come_from_coordinates(labels, expected_unit):
    rf = build_resolved_frame(
        frame_ref=_FRAME_REF,
        role_keys=("recipient",),
        rows=[_row(period="2020", unit="unit/USD", provenance="src_oecd")],
        labels=labels,
    )
    (series,) = resolve_selectors(
        resolved_frame=rf,
        selectors=[
            {
                "bindings": [
                    {"dimension": "recipient", "values": ["country/ETH"]},
                    {"dimension": "unit", "values": ["unit/USD"]},
                ]
            }
        ],
        window={"start": "2020", "end": "2020"},
    )
    bound = bind_series(resolved_frame=rf, series=series, window={"start": "2020", "end": "2020"})
    assert bound.unit == expected_unit
    assert bound.source_ref == "src_oecd"


def test_converted_unit_passes_through_as_the_raw_structured_object():
    """A JSON unit request reaches `BoundSeries.unit` unchanged.

    Build the frame dict directly because `Frame.build` stores a `UnitRequest`
    object, while real frame JSON carries a dict. Coordinates use rendered text.
    """
    doc = {
        "columns": ["stat_var", "recipient", "unit", "time_period", "value"],
        "rows": [
            ["dcid:X", "country/ETH", {"currency": "USD", "prices": "current"}, "2020", 442.0],
        ],
    }
    rf = ResolvedFrame(
        frame_ref=_FRAME_REF,
        frame=doc,
        structural_view=compute_structural_view(frame=doc),
        manifest={},
        labels={},
        retrieved_on=FIXTURE_RETRIEVED_ON,
    )
    (series,) = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]}],
        window={"start": "2020", "end": "2020"},
    )
    assert dict(series.coordinates)["unit"] == "USD/current"
    bound = bind_series(resolved_frame=rf, series=series, window={"start": "2020", "end": "2020"})
    assert bound.unit == {"currency": "USD", "prices": "current"}


@pytest.mark.parametrize(
    ("part", "labels", "expected"),
    [
        pytest.param("USDollar", {"USDollar": "US Dollar"}, "US Dollar", id="labelled-ref"),
        pytest.param("USDollar", {}, "USDollar", id="unlabelled-ref-stays-raw"),
        pytest.param("USDollar", {"USDollar": "U" * 201}, "U" * 200, id="long-ref-label-is-cut"),
        pytest.param(
            {"stat_var": "dcid:Count_Person"},
            {"dcid:Count_Person": "P" * 201},
            {"stat_var": "dcid:Count_Person", "label": "P" * 200},
            id="long-stat-var-label-is-cut",
        ),
        pytest.param(
            {"currency": "USD", "prices": "constant", "base_year": 2023},
            {},
            {"currency": "USD", "prices": "constant", "base_year": 2023},
            id="converted-unit-passes-through",
        ),
        pytest.param(
            {"stat_var": "dcid:Count_Person"},
            {"dcid:Count_Person": "Total population"},
            {"stat_var": "dcid:Count_Person", "label": "Total population"},
            id="labelled-stat-var",
        ),
        pytest.param(
            {"stat_var": "dcid:Count_Person"},
            {},
            {"stat_var": "dcid:Count_Person", "label": "Count_Person"},
            id="unlabelled-stat-var-takes-its-bare-ref",
        ),
    ],
)
def test_composed_unit_binds_each_side_as_a_catalogue_unit_part(part, labels, expected):
    rf = _stored_resolved_frame(
        frame_ref=_FRAME_REF,
        role_keys=("recipient",),
        rows=[_row(period="2020", unit={"numerator": part, "denominator": part})],
        labels=labels,
    )
    (series,) = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]}],
        window={"start": "2020", "end": "2020"},
    )
    bound = bind_series(resolved_frame=rf, series=series, window={"start": "2020", "end": "2020"})
    assert bound.unit == {"numerator": expected, "denominator": expected}


def test_label_prefers_the_labels_map():
    rf = build_resolved_frame(
        frame_ref=_FRAME_REF,
        role_keys=("recipient",),
        rows=[_row(period="2020")],
        labels={"country/ETH": "Ethiopia"},
    )
    (series,) = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]}],
        window={"start": "2020", "end": "2020"},
    )
    bound = bind_series(resolved_frame=rf, series=series, window={"start": "2020", "end": "2020"})
    assert bound.label == "Ethiopia"


def test_label_falls_back_to_raw_ref_when_unlabeled():
    rf = build_resolved_frame(
        frame_ref=_FRAME_REF, role_keys=("recipient",), rows=[_row(period="2020")], labels={}
    )
    (series,) = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]}],
        window={"start": "2020", "end": "2020"},
    )
    bound = bind_series(resolved_frame=rf, series=series, window={"start": "2020", "end": "2020"})
    assert bound.label == "country/ETH"


def test_month_grain_period_sequence():
    doc = {
        "frame_version": "frame1",
        "columns": [
            "stat_var",
            "recipient",
            "time_period",
            "value",
            "unit",
            "measurementMethod",
            "provenance",
            "status",
        ],
        "rows": [
            ["dcid:X", "country/ETH", "2020-11", 1.0, None, None, "src_a", None],
            ["dcid:X", "country/ETH", "2021-01", 2.0, None, None, "src_a", None],
        ],
    }
    rf = ResolvedFrame(
        frame_ref=_FRAME_REF,
        frame=doc,
        structural_view=compute_structural_view(frame=doc),
        manifest={},
        labels={},
        retrieved_on=FIXTURE_RETRIEVED_ON,
    )
    (series,) = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]}],
        window={"start": "2020-11", "end": "2021-01"},
    )
    bound = bind_series(
        resolved_frame=rf, series=series, window={"start": "2020-11", "end": "2021-01"}
    )
    periods = [p["period"] for p in bound.points]
    assert periods == ["2020-11", "2020-12", "2021-01"]


def test_count_periods_day_grain_clamps_a_calendar_invalid_day_instead_of_raising():
    # The grammar accepts source dates such as 2024-02-30; leap-year arithmetic clamps it to Feb 29.
    assert count_periods(start="2024-02-01", end="2024-02-30", grain="day") == 29


def test_count_periods_day_grain_clamps_a_non_leap_february():
    assert count_periods(start="2023-02-01", end="2023-02-30", grain="day") == 28


def test_day_grain_binds_a_calendar_invalid_observed_day_alongside_the_calendar_backbone():
    """The observed grammar-valid date `2024-02-30` joins the calendar backbone.

    The unobserved `2024-02-29` remains a typed missing point, and the union sorts lexically.
    """
    rf = build_resolved_frame(
        frame_ref=_FRAME_REF,
        role_keys=("recipient",),
        rows=[
            _row(period="2024-02-28", value=1.0),
            _row(period="2024-02-30", value=2.0),
            _row(period="2024-03-01", value=3.0),
        ],
    )
    (series,) = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]}],
        window={"start": "2024-02-28", "end": "2024-03-01"},
    )
    bound = bind_series(
        resolved_frame=rf, series=series, window={"start": "2024-02-28", "end": "2024-03-01"}
    )
    periods = [p["period"] for p in bound.points]
    assert periods == ["2024-02-28", "2024-02-29", "2024-02-30", "2024-03-01"]
    points_by_period = {p["period"]: p for p in bound.points}
    assert points_by_period["2024-02-28"] == {"period": "2024-02-28", "value": 1.0}
    assert points_by_period["2024-02-29"] == {"period": "2024-02-29", "status": "missing"}
    assert points_by_period["2024-02-30"] == {"period": "2024-02-30", "value": 2.0}
    assert points_by_period["2024-03-01"] == {"period": "2024-03-01", "value": 3.0}

    # The cap check must count the same four-point observed/calendar union as binding.
    count = count_periods(
        start="2024-02-28",
        end="2024-03-01",
        grain="day",
        observed=frame_periods(resolved_frame=rf),
    )
    assert count == len(bound.points) == 4


def test_count_periods_day_grain_never_undercounts_an_observed_non_calendar_day():
    """Include observed non-calendar dates in the cap count.

    This window binds four points, one more than its three calendar days.
    """
    observed = frozenset({"2024-02-28", "2024-02-30", "2024-03-01"})
    assert count_periods(start="2024-02-28", end="2024-03-01", grain="day", observed=observed) == 4


def test_period_sequence_never_includes_a_point_lexically_before_a_calendar_invalid_start():
    """Clamping `start="2024-02-30"` yields `2024-02-29`; a one-day window must exclude it."""
    rf = build_resolved_frame(
        frame_ref=_FRAME_REF,
        role_keys=("recipient",),
        rows=[_row(period="2024-02-30", value=1.0)],
    )
    (series,) = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]}],
        window={"start": "2024-02-30", "end": "2024-02-30"},
    )
    bound = bind_series(
        resolved_frame=rf, series=series, window={"start": "2024-02-30", "end": "2024-02-30"}
    )
    periods = [p["period"] for p in bound.points]
    assert periods == ["2024-02-30"]

    count = count_periods(
        start="2024-02-30",
        end="2024-02-30",
        grain="day",
        observed=frame_periods(resolved_frame=rf),
    )
    assert count == len(bound.points) == 1


def test_day_grain_observed_union_excludes_a_non_calendar_day_outside_the_window():
    """Only observed non-calendar dates inside the window join its period sequence."""
    rf = build_resolved_frame(
        frame_ref=_FRAME_REF,
        role_keys=("recipient",),
        rows=[
            _row(period="2024-02-15", value=1.0),
            _row(period="2024-02-30", value=2.0),
        ],
    )
    (series,) = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]}],
        window={"start": "2024-02-14", "end": "2024-02-16"},
    )
    bound = bind_series(
        resolved_frame=rf, series=series, window={"start": "2024-02-14", "end": "2024-02-16"}
    )
    periods = [p["period"] for p in bound.points]
    assert periods == ["2024-02-14", "2024-02-15", "2024-02-16"]
    count = count_periods(
        start="2024-02-14",
        end="2024-02-16",
        grain="day",
        observed=frame_periods(resolved_frame=rf),
    )
    assert count == len(bound.points) == 3


def test_count_periods_matches_bound_length_with_an_invalid_end():
    """A calendar-invalid end clamps to Feb 29 for arithmetic; the observed Feb 30 adds a point."""
    rf = build_resolved_frame(
        frame_ref=_FRAME_REF,
        role_keys=("recipient",),
        rows=[
            _row(period="2024-02-27", value=1.0),
            _row(period="2024-02-30", value=2.0),
        ],
    )
    (series,) = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]}],
        window={"start": "2024-02-27", "end": "2024-02-30"},
    )
    bound = bind_series(
        resolved_frame=rf, series=series, window={"start": "2024-02-27", "end": "2024-02-30"}
    )
    periods = [p["period"] for p in bound.points]
    assert periods == ["2024-02-27", "2024-02-28", "2024-02-29", "2024-02-30"]
    count = count_periods(
        start="2024-02-27",
        end="2024-02-30",
        grain="day",
        observed=frame_periods(resolved_frame=rf),
    )
    assert count == len(bound.points) == 4


def test_count_periods_matches_bound_length_with_both_endpoints_invalid():
    """Invalid endpoints `2024-02-30` and `2024-04-31` add to the 61 real days between them."""
    rf = build_resolved_frame(
        frame_ref=_FRAME_REF,
        role_keys=("recipient",),
        rows=[
            _row(period="2024-02-30", value=1.0),
            _row(period="2024-04-31", value=2.0),
        ],
    )
    (series,) = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]}],
        window={"start": "2024-02-30", "end": "2024-04-31"},
    )
    bound = bind_series(
        resolved_frame=rf, series=series, window={"start": "2024-02-30", "end": "2024-04-31"}
    )
    periods = [p["period"] for p in bound.points]
    assert periods[0] == "2024-02-30"
    assert periods[1] == "2024-03-01"
    assert periods[-2] == "2024-04-30"
    assert periods[-1] == "2024-04-31"
    assert len(periods) == 63  # 61 real calendar days (Mar 1..Apr 30) + the 2 invalid endpoints

    count = count_periods(
        start="2024-02-30",
        end="2024-04-31",
        grain="day",
        observed=frame_periods(resolved_frame=rf),
    )
    assert count == len(bound.points) == 63


def test_count_periods_matches_bound_length_with_invalid_days_inside_and_outside_the_window():
    """One observed non-calendar day falls inside `[start, end]` (adds to
    the count); another falls outside it (must not)."""
    rf = build_resolved_frame(
        frame_ref=_FRAME_REF,
        role_keys=("recipient",),
        rows=[
            _row(period="2024-02-30", value=1.0),  # inside
            _row(period="2024-03-01", value=2.0),  # inside
            _row(period="2024-04-31", value=3.0),  # outside
        ],
    )
    (series,) = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]}],
        window={"start": "2024-02-28", "end": "2024-03-02"},
    )
    bound = bind_series(
        resolved_frame=rf, series=series, window={"start": "2024-02-28", "end": "2024-03-02"}
    )
    periods = [p["period"] for p in bound.points]
    assert periods == ["2024-02-28", "2024-02-29", "2024-02-30", "2024-03-01", "2024-03-02"]

    count = count_periods(
        start="2024-02-28",
        end="2024-03-02",
        grain="day",
        observed=frame_periods(resolved_frame=rf),
    )
    assert count == len(bound.points) == 5


def test_count_periods_day_grain_never_enumerates_a_multi_millennium_window():
    """Count the full year 0001-9999 window without enumerating each day.

    The user-authored window has no span bound, so counting must use closed-form date arithmetic.
    """
    start, end = "0001-01-01", "9999-12-31"
    expected = (datetime.date(9999, 12, 31) - datetime.date(1, 1, 1)).days + 1
    assert count_periods(start=start, end=end, grain="day") == expected
