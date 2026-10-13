"""Checks date and source tokens derived from resolved answer fields."""

from __future__ import annotations

import pytest

from search.models import (
    BindingUnbound,
    BindingValue,
    DateRange,
    ObservationPoint,
    ObservationSeries,
    ProvenanceRef,
    SeriesEntityRef,
    SeriesStatVarRef,
    Slot,
    SlotKey,
    SlotValue,
    SpecObservations,
    StatVar,
    TimeWindow,
)
from search.render.plan import render_sentence, render_sentence_plan
from search.render.tokens import sentence_from_plan
from tests.render._helpers import _ref, _spec, _unbound_source_slot, assert_round_trip

# ---------------------------------------------------------------------------
# Case 20: Resolved when/source tokens once the answer's own facet
# dates and provenance are known, for a query that left when/source unbound.
# ---------------------------------------------------------------------------


def _unbound_when_slot() -> Slot:
    return Slot(
        key=SlotKey(axis="when", property=None, label="time window"), binding=BindingUnbound()
    )


def _sv_with_range(dcid: str, label: str, start: str | None, end: str | None) -> StatVar:
    return StatVar(
        ref=_ref(dcid, label),
        shape_id="test-shape",
        slot_values=[],
        constraint_properties=[],
        data_date_range=DateRange(start=start, end=end),
    )


def test_when_token_resolves_to_the_single_served_year():
    """A single served year reads "in" that year."""
    spec = _spec(
        slots=[_unbound_when_slot()],
        stat_vars=[_sv_with_range("sv1", "SV1", "2024", "2024")],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    when_tok = next(t for t in plan if t.axis == "when")
    assert when_tok.text == "in 2024"
    assert when_tok.binding_ref == "when:"
    assert sentence_from_plan(plan) == "Development finance flow in 2024."


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        pytest.param("2004", "2024", "2004 to 2024", id="year-range"),
        pytest.param("2015-06", "2020-01", "June 2015 to January 2020", id="month-range"),
        pytest.param("2006-04-02", "2018", "2 April 2006 to 2018", id="day-and-year-range"),
        pytest.param("2020Q1", "2021", "2020Q1 to 2021", id="unrecognised-date-verbatim"),
    ],
)
def test_when_token_formats_two_bound_ranges(start: str, end: str, expected: str):
    """Format confirmed date bounds at their supplied precision."""
    spec = _spec(
        slots=[_unbound_when_slot()],
        stat_vars=[_sv_with_range("sv1", "SV1", start, end)],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    when_tok = next(t for t in plan if t.axis == "when")
    assert when_tok.text == expected


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        pytest.param("2022", "in 2022", id="year"),
        pytest.param("2025-06", "in June 2025", id="month"),
    ],
)
def test_when_token_formats_equal_date_as_one_period(date: str, expected: str):
    """Equal bounds read "in" that date at its precision."""
    spec = _spec(
        slots=[_unbound_when_slot()],
        stat_vars=[_sv_with_range("sv1", "SV1", date, date)],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    when_tok = next(t for t in plan if t.axis == "when")
    assert when_tok.text == expected


def test_when_token_resolves_to_since_form_when_only_start_confirmed():
    """A start without an end is reported as `since`."""
    spec = _spec(
        slots=[_unbound_when_slot()],
        stat_vars=[_sv_with_range("sv1", "SV1", "2015", None)],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    when_tok = next(t for t in plan if t.axis == "when")
    assert when_tok.text == "since 2015"
    assert sentence_from_plan(plan) == "Development finance flow, since 2015."


def test_when_token_resolves_to_until_form_when_only_end_confirmed():
    """An end without a start is reported as `until`."""
    spec = _spec(
        slots=[_unbound_when_slot()],
        stat_vars=[_sv_with_range("sv1", "SV1", None, "2020")],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    when_tok = next(t for t in plan if t.axis == "when")
    assert when_tok.text == "until 2020"


def test_when_token_spans_every_stat_var_range():
    """Different StatVar date ranges read as one span, earliest start to latest end."""
    spec = _spec(
        slots=[_unbound_when_slot()],
        stat_vars=[
            _sv_with_range("sv1", "SV1", "2024", "2024"),
            _sv_with_range("sv2", "SV2", "2020", "2020"),
        ],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    when_tok = next(t for t in plan if t.axis == "when")
    assert when_tok.text == "2020 to 2024"


def test_when_token_stays_empty_with_no_confirmed_date_range():
    """Without confirmed dates, the when token stays empty."""
    spec = _spec(slots=[_unbound_when_slot()])
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    when_tok = next(t for t in plan if t.axis == "when")
    assert when_tok.text == ""


def test_when_token_reads_the_served_range_before_the_coverage_window():
    """The StatVars' served ranges state the period; the coverage window, in years only,
    states it when no StatVar has a range."""
    served = _spec(
        slots=[_unbound_when_slot()],
        window=TimeWindow(start_year=2020, end_year=2020),
        stat_vars=[_sv_with_range("sv1", "SV1", "2020-03", "2020-03")],
    )
    plan = render_sentence_plan(served)
    assert_round_trip(plan, render_sentence(served))
    assert next(t for t in plan if t.axis == "when").text == "in March 2020"

    window_only = _spec(slots=[_unbound_when_slot()], window=TimeWindow(start_year=2020))
    plan = render_sentence_plan(window_only)
    assert next(t for t in plan if t.axis == "when").text == "since 2020"


def test_source_token_resolves_single_confirmed_provenance():
    """Resolved provenance renders without editable slot metadata."""
    spec = _spec(
        slots=[_unbound_source_slot()],
        resolved_sources=[_ref("worldBank/WDI", "World Bank WDI")],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    source_tok = next(t for t in plan if t.axis == "source")
    assert source_tok.text == "World Bank WDI"
    assert source_tok.binding_ref is None
    assert source_tok.property_dcid is None
    assert source_tok.segment == "byline"
    assert sentence_from_plan(plan) == "Development finance flow. World Bank WDI."


def test_source_token_names_two_resolved_sources():
    """Join two resolved provenance labels with `and`."""
    spec = _spec(
        slots=[_unbound_source_slot()],
        resolved_sources=[
            _ref("dc/base/WDI", "WorldDevelopmentIndicators"),
            _ref("dc/base/Kenya_Census", "Kenya_Census"),
        ],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    source_tok = next(t for t in plan if t.axis == "source")
    assert source_tok.text == "World Development Indicators and Kenya Census"
    assert sentence_from_plan(plan) == (
        "Development finance flow. World Development Indicators and Kenya Census."
    )


def test_source_token_names_three_resolved_sources():
    """Join three resolved provenance labels with an Oxford comma."""
    spec = _spec(
        slots=[_unbound_source_slot()],
        resolved_sources=[
            _ref("worldBank/WDI", "World Bank WDI"),
            _ref("un/Pop", "UN Population Division"),
            _ref("dc/base/Kenya_Census", "Kenya_Census"),
        ],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    source_tok = next(t for t in plan if t.axis == "source")
    assert source_tok.text == "World Bank WDI, UN Population Division, and Kenya Census"


def test_source_token_collapses_four_resolved_sources_to_a_count():
    """Four or more provenances render as a count."""
    spec = _spec(
        slots=[_unbound_source_slot()],
        resolved_sources=[
            _ref("worldBank/WDI", "World Bank WDI"),
            _ref("un/Pop", "UN Population Division"),
            _ref("dc/base/Kenya_Census", "Kenya_Census"),
            _ref("oecd/Reg", "OECD Regional Demography Life Expectancy"),
        ],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    source_tok = next(t for t in plan if t.axis == "source")
    assert source_tok.text == "4 sources"
    assert sentence_from_plan(plan) == "Development finance flow. 4 sources."


def test_source_token_stays_empty_with_no_resolved_sources():
    """Without resolved sources, the source token stays empty."""
    spec = _spec(slots=[_unbound_source_slot()])
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    source_tok = next(t for t in plan if t.axis == "source")
    assert source_tok.text == ""


def test_source_token_bound_slot_wins_over_resolved_fallback():
    """A bound source slot takes precedence and retains its binding reference."""
    source_slot = Slot(
        key=SlotKey(axis="source", property=None, label="source"),
        binding=BindingValue(value=SlotValue(ref=_ref("OECDsrc", "OECD"), value_kind="source")),
    )
    spec = _spec(
        slots=[source_slot],
        resolved_sources=[_ref("worldBank/WDI", "World Bank WDI")],
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    source_tok = next(t for t in plan if t.axis == "source")
    assert source_tok.text == "OECD"
    assert source_tok.binding_ref == "source:"


def test_resolved_when_and_source_tokens_never_carry_the_observation_value():
    """Resolved date and source tokens omit observation values."""
    distinctive_value = 24_014_716.0
    spec = _spec(
        slots=[_unbound_when_slot(), _unbound_source_slot()],
        stat_vars=[_sv_with_range("sv1", "Total population", "2019", "2019")],
        resolved_sources=[_ref("worldBank/WDI", "World Bank WDI")],
        observations=SpecObservations(
            series=[
                ObservationSeries(
                    stat_var=SeriesStatVarRef(ref="sv1"),
                    entities=[SeriesEntityRef(role="observationAbout", ref="country/KEN")],
                    points=[ObservationPoint(date="2019", value=distinctive_value)],
                    total_obs=1,
                    source_refs=[ProvenanceRef(ref="worldBank/WDI", label="World Bank WDI")],
                )
            ],
            total_obs=1,
            truncated=False,
        ),
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    sentence = sentence_from_plan(plan)
    assert "in 2019" in sentence
    assert "World Bank WDI" in sentence
    assert str(distinctive_value) not in sentence
    assert "24014716" not in sentence.replace(",", "").replace(".", "")
    for tok in plan:
        assert str(distinctive_value) not in tok.text


def _served(dates: list[str], *, truncated: bool = False) -> SpecObservations:
    return SpecObservations(
        series=[
            ObservationSeries(
                stat_var=SeriesStatVarRef(ref="sv1"),
                entities=[SeriesEntityRef(role="destination", ref="country/KEN")],
                points=[ObservationPoint(date=date, value=1.0) for date in dates],
                total_obs=len(dates),
            )
        ],
        total_obs=len(dates),
        truncated=truncated,
    )


@pytest.mark.parametrize(
    "observations",
    [
        _served([str(year) for year in range(1970, 2025)]),
        _served(["2023", "2024"], truncated=True),
    ],
    ids=["delivered-part", "truncated"],
)
def test_when_token_reads_the_served_range_whatever_the_preview_delivers(observations):
    """The StatVar's range is the dates its read plan serves, 1970 to 2032. A preview that
    delivers part of them, or a truncated one, leaves the text as it reads without one."""
    spec = _spec(
        slots=[_unbound_when_slot()],
        stat_vars=[_sv_with_range("sv1", "SV1", "1970", "2032")],
        observations=observations,
    )
    plan = render_sentence_plan(spec)
    assert_round_trip(plan, render_sentence(spec))
    when_tok = next(t for t in plan if t.axis == "when")
    assert when_tok.text == "1970 to 2032"


def test_when_token_keeps_facet_dates_without_delivered_points():
    spec = _spec(
        slots=[_unbound_when_slot()],
        stat_vars=[_sv_with_range("sv1", "SV1", "1970", "2032")],
        observations=_served([]),
    )
    plan = render_sentence_plan(spec)
    when_tok = next(t for t in plan if t.axis == "when")
    assert when_tok.text == "1970 to 2032"
