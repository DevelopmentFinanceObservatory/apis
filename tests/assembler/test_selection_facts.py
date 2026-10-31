"""Pin a card's selection facts to the series it draws, built through the write path."""

from __future__ import annotations

import datetime

import pytest

from assembler.fakes import FakeFrameStore
from assembler.mint.selection import selection_facts
from assembler.pipeline.candidate import _build_candidate
from assembler.ports import FrameRef
from assembler.testing import malaria_ethiopia_frame_ref, malaria_ethiopia_resolved_frame
from description.facts import Coordinate, SelectionFacts, Span
from description.steps import ComposedUnit, SimpleUnit, VariableSide

from ._pipeline_support import _multi_recipient_frame, _stored_resolved_frame

_FRAME_REF = FrameRef(spec_id="spec_" + "c" * 32, content_version="cv_" + "d" * 32)


def _clock() -> datetime.datetime:
    return datetime.datetime(2026, 10, 11, tzinfo=datetime.UTC)


def _facts(*, release, frame, intent) -> SelectionFacts:
    """Builds the card as a create does, then computes its facts from what the candidate
    holds."""
    candidate = _build_candidate(
        patch={
            **intent,
            "frame_ref": {
                "spec_id": frame.frame_ref.spec_id,
                "content_version": frame.frame_ref.content_version,
            },
        },
        current=None,
        is_create=True,
        release=release,
        frame_store=FakeFrameStore(frames={frame.frame_ref: frame}),
        clock=_clock,
    )
    assert candidate.resolved_frame is not None
    return selection_facts(
        merged_intent=candidate.stored_intent,
        resolved_frame=candidate.resolved_frame,
        series=candidate.series,
        bound_series=candidate.bound_series,
    )


def _row(*, stat_var, period, value=1.0, unit=None):
    return {
        "stat_var": stat_var,
        "recipient": "country/ETH",
        "time_period": period,
        "value": value,
        "unit": unit,
        "measurementMethod": None,
        "provenance": "src_oecd_crs",
    }


def _paired_frame(*, health, education):
    """Two variables over the same two sectors and two flow types, paired as named."""
    return _stored_resolved_frame(
        frame_ref=_FRAME_REF,
        role_keys=("recipient",),
        rows=[
            _row(stat_var=stat_var, period=period)
            for stat_var in (health, education)
            for period in ("2020", "2021")
        ],
    )


def test_same_dimension_totals_paired_apart_give_different_facts_for_a_health_card(release):
    health_grant = _paired_frame(health="dcid:Health_Grant", education="dcid:Education_Loan")
    health_loan = _paired_frame(health="dcid:Health_Loan", education="dcid:Education_Grant")

    facts = [
        _facts(
            release=release,
            frame=frame,
            intent={
                "type": "line_chart",
                "select": {"stat_var": health, "recipient": "country/ETH"},
            },
        )
        for frame, health in (
            (health_grant, "dcid:Health_Grant"),
            (health_loan, "dcid:Health_Loan"),
        )
    ]

    assert facts[0] != facts[1]
    assert [[c.value for c in one.series[0].coordinates] for one in facts] == [
        ["Health_Grant", "country/ETH", None],
        ["Health_Loan", "country/ETH", None],
    ]
    assert [c.column for c in facts[0].series[0].coordinates] == [
        "stat_var",
        "recipient",
        "measurementMethod",
    ]
    assert facts[0].series[0].source == "src_oecd_crs"


@pytest.mark.parametrize(
    ("window", "first", "last", "values", "missing"),
    [
        pytest.param({"start": "2015", "end": "2020"}, "2015", "2020", 5, 1, id="ends-before-2024"),
        pytest.param(
            {"start": "2016", "end": "2018"}, "2016", "2017", 2, 1, id="end-holds-no-value"
        ),
    ],
)
def test_periods_and_counts_stay_inside_the_card_window(
    release, window, first, last, values, missing
):
    # The malaria series runs 2015 to 2024 and holds no 2018 value.
    facts = _facts(
        release=release,
        frame=malaria_ethiopia_resolved_frame(),
        intent={
            "type": "table",
            "select": {"stat_var": "dcid:MalariaGrantDisbursements", "recipient": "country/ETH"},
            "window": window,
        },
    )

    assert facts.window == Span(**window)
    (drawn,) = facts.series
    assert (drawn.first_period, drawn.last_period, drawn.values, drawn.missing) == (
        first,
        last,
        values,
        missing,
    )


def test_status_counts_are_unknown_not_zero(release):
    facts = _facts(
        release=release,
        frame=malaria_ethiopia_resolved_frame(),
        intent={
            "type": "stat_card",
            "select": {"stat_var": "dcid:MalariaGrantDisbursements", "recipient": "country/ETH"},
        },
    )

    (drawn,) = facts.series
    assert drawn.status.model_dump() == {"estimated": None, "provisional": None}
    # A stat card names no window, so its window is the frame's coverage.
    assert facts.window == Span(start="2015", end="2024")


_GDP = "dcid:Amount_EconomicActivity_GrossDomesticProduction_Nominal"


def _weo_frame(*, estimated_from: int):
    """IMF WEO GDP for Ethiopia from 2016 to 2030: reported with no method before
    `estimated_from`, estimated from it on."""
    return _stored_resolved_frame(
        frame_ref=_FRAME_REF,
        role_keys=("recipient",),
        rows=[
            _row(stat_var=_GDP, period=str(year))
            | {
                "measurementMethod": None if year < estimated_from else "Estimated",
                "provenance": "provenance/IMFWorldEconomicOutlook",
            }
            for year in range(2016, 2031)
        ],
    )


@pytest.mark.parametrize(
    ("estimated_from", "method", "window", "estimated"),
    [
        pytest.param(2025, "Estimated", ("2016", "2030"), 6, id="estimates-over-the-span"),
        pytest.param(2025, "Estimated", ("2020", "2026"), 2, id="estimates-inside-the-window"),
        pytest.param(2031, None, ("2016", "2030"), 0, id="reported-under-weo"),
    ],
)
def test_status_counts_are_the_drawn_points_counts(
    release, estimated_from, method, window, estimated
):
    select = {"stat_var": _GDP, "recipient": "country/ETH"}
    facts = _facts(
        release=release,
        frame=_weo_frame(estimated_from=estimated_from),
        intent={
            "type": "line_chart",
            "select": select | ({"measurementMethod": method} if method else {}),
            "window": {"start": window[0], "end": window[1]},
        },
    )

    (drawn,) = facts.series
    assert drawn.status.model_dump() == {"estimated": estimated, "provisional": 0}


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        pytest.param(
            {"currency": "USD", "prices": "constant", "base_year": 2023},
            SimpleUnit(currency="USD", prices="constant", base_year=2023),
            id="constant-prices-keep-base-year",
        ),
        pytest.param(
            {"currency": "EUR", "prices": "current"},
            SimpleUnit(currency="EUR", prices="current"),
            id="current-prices",
        ),
        pytest.param(
            {"numerator": "USDollar", "denominator": {"stat_var": "dcid:Count_Person"}},
            ComposedUnit(
                numerator=SimpleUnit(ref="USDollar", label="US dollars"),
                denominator=VariableSide(stat_var="Count_Person", label="Population"),
            ),
            id="composed-variable-side-by-bare-dcid",
        ),
        pytest.param(
            {"numerator": "USDollar", "denominator": {"stat_var": "dcid:Count_Household"}},
            ComposedUnit(
                numerator=SimpleUnit(ref="USDollar", label="US dollars"),
                denominator=VariableSide(stat_var="Count_Household"),
            ),
            id="composed-unlabelled-variable-side",
        ),
        pytest.param(
            {
                "numerator": {"currency": "USD", "prices": "constant", "base_year": 2020},
                "denominator": "Percent",
            },
            ComposedUnit(
                numerator=SimpleUnit(currency="USD", prices="constant", base_year=2020),
                denominator=SimpleUnit(ref="Percent"),
            ),
            id="composed-converted-side",
        ),
        pytest.param("USDollar", SimpleUnit(ref="USDollar", label="US dollars"), id="ref"),
        pytest.param(None, None, id="no-unit"),
    ],
)
def test_unit_maps_to_the_kit_shape(release, cell, expected):
    frame = _stored_resolved_frame(
        frame_ref=_FRAME_REF,
        role_keys=("recipient",),
        rows=[_row(stat_var="dcid:X", period="2020", unit=cell)],
        labels={"USDollar": "US dollars", "dcid:Count_Person": "Population"},
    )

    facts = _facts(
        release=release,
        frame=frame,
        intent={"type": "stat_card", "select": {"stat_var": "dcid:X", "recipient": "country/ETH"}},
    )

    assert facts.series[0].unit == expected


def test_series_follow_the_select_in_one_order_whatever_order_it_lists(release):
    def draw(recipients):
        return _facts(
            release=release,
            frame=_multi_recipient_frame(),
            intent={
                "type": "line_chart",
                "select": {"stat_var": "dcid:MalariaGrantDisbursements", "recipient": recipients},
                "window": {"start": "2020", "end": "2022"},
            },
        )

    forward = draw(["country/ETH", "country/GHA", "country/KEN"])
    backward = draw(["country/KEN", "country/GHA", "country/ETH"])

    assert forward == draw(["country/ETH", "country/GHA", "country/KEN"])
    assert [
        Coordinate(column="recipient", value=f"country/{iso}") in one.coordinates
        for one, iso in zip(forward.series, ("ETH", "GHA", "KEN"), strict=True)
    ] == [True, True, True]
    assert backward.series == forward.series


def test_a_source_card_draws_no_series_and_has_no_facts(release):
    frame = malaria_ethiopia_resolved_frame()
    candidate = _build_candidate(
        patch={
            "type": "source_card",
            "frame_ref": {
                "spec_id": frame.frame_ref.spec_id,
                "content_version": frame.frame_ref.content_version,
            },
        },
        current=None,
        is_create=True,
        release=release,
        frame_store=FakeFrameStore(frames={malaria_ethiopia_frame_ref(): frame}),
        clock=_clock,
    )
    assert candidate.resolved_frame is not None

    with pytest.raises(ValueError, match="draws no series"):
        selection_facts(
            merged_intent=candidate.stored_intent,
            resolved_frame=candidate.resolved_frame,
            series=candidate.series,
            bound_series=candidate.bound_series,
        )
