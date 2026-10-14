"""Acceptance cases for a window that runs past the elected series (decision 15, #1047).

A GDP request for every country from 2016 to 2030 returns the IMF's estimates for the years
after its reported data. The same source's `Estimated` series continues the elected series in
the elected unit, each row keeps its `measurementMethod`, and the reported value wins where
both hold a year.

The data is WEO nominal GDP as the dev graph holds it (`_weo_recorded`): reported values end
in 2025 at France and Ecuador and in 2024 at Kenya and Zimbabwe, and the estimates start the
year after. The overlap row adds one constructed estimate, since the recorded series do not
overlap.

Estimates are counted in the mint's series figures through the method classification (P1)
and, beside it, a declared list of provenances whose method-less rows count as reported (P34).
IMF WEO is on that list, so a window mixing its reported rows with its estimates counts the
estimates and no provisional values. A provenance off the list keeps both counts unknown, since
its method-less rows say nothing of their status.
"""

from __future__ import annotations

from typing import cast

import pytest

from materialise import OverCeilingError
from materialise.facets import FacetBinding, UnitRequest
from materialise.graph import Facet
from materialise.recipes import RateCoverageGapError
from materialise.window import RequestedWindow

from ._weo_recorded import (
    ECUADOR,
    ESTIMATED,
    FRANCE,
    GDP,
    KENYA,
    PLACES,
    WEO,
    ZIMBABWE,
    Values,
    facet,
    facet_id,
    mint,
    mint_group,
    preview,
    recorded_facets,
    recorded_values,
    rows,
)

_USD = "CurrencyUSD"
_WINDOW = RequestedWindow(kind="range", start_year=2016, end_year=2030)
_LAST_REPORTED = {FRANCE: 2025, KENYA: 2024, ZIMBABWE: 2024, ECUADOR: 2025}
"""The last year each place's reported series holds, as recorded."""


def _expected(values: Values, *, overlap: dict[tuple[str, str], str] | None = None) -> dict:
    """Each served `(place, year)`: its value and method, read from the recorded US dollar
    series. A reported year takes the reported value; a later year the estimate."""
    expected: dict[tuple[str, str], tuple[float, str | None]] = {}
    for place in PLACES:
        reported = facet_id(place, unit=_USD, method=None)
        estimated = facet_id(place, unit=_USD, method=ESTIMATED)
        for year in range(2016, 2031):
            key = str(year)
            if year <= _LAST_REPORTED[place]:
                expected[(place, key)] = (values[(place, reported)][key], None)
            else:
                expected[(place, key)] = (values[(place, estimated)][key], ESTIMATED)
    return expected | dict(overlap or {})


def _served(result) -> dict[tuple[str, str], tuple[float, str | None]]:
    served: dict[tuple[str, str], tuple[float, str | None]] = {}
    for row in rows(result):
        key = (str(row["observationAbout"]), str(row["time_period"]))
        assert key not in served, f"{key} is served twice"
        served[key] = (float(row["value"]), row["measurementMethod"])
    return served


def test_a_gdp_request_from_2016_to_2030_serves_the_imfs_estimates_after_its_reported_data():
    """Case 10. Every place gets fifteen years in US dollars from the IMF: reported values
    through its last reported year with no method, then the estimates, each marked
    `Estimated`."""
    result = preview(PLACES, window=_WINDOW)

    assert result.selections["measurementMethod"].continued_by == ESTIMATED
    assert result.selections["provenance"].ref == WEO
    assert result.selections["unit"].ref == _USD
    assert _served(result) == _expected(recorded_values())
    assert {row["unit"] for row in rows(result)} == {_USD}
    assert {row["provenance"] for row in rows(result)} == {WEO}


def _kenya_overlap() -> tuple[dict[str, list[Facet]], Values]:
    """The recorded facets and values with one constructed estimate for Kenya in 2024, a year
    its reported series holds."""
    estimate = facet(KENYA, unit=_USD, method=ESTIMATED)
    overlapping = Facet(
        facet_id=estimate.facet_id,
        provenance_id=estimate.provenance_id,
        measurement_method=estimate.measurement_method,
        unit=estimate.unit,
        obs_count=(estimate.obs_count or 0) + 1,
        earliest_date="2024",
        latest_date=estimate.latest_date,
    )
    facets_by_place = recorded_facets()
    facets_by_place[KENYA] = [
        overlapping if candidate == estimate else candidate for candidate in facets_by_place[KENYA]
    ]
    values = recorded_values()
    estimated = facet_id(KENYA, unit=_USD, method=ESTIMATED)
    values[(KENYA, estimated)] = {"2024": 1.0, **values[(KENYA, estimated)]}
    return facets_by_place, values


def test_the_reported_value_wins_a_year_both_series_hold():
    """A constructed estimate for Kenya in 2024, a year its reported series holds, is not
    served: 2024 keeps the reported value and no method, and the trace's continuation starts
    in 2025."""
    facets_by_place, values = _kenya_overlap()

    result = preview(PLACES, window=_WINDOW, facets_by_place=facets_by_place, values=values)

    assert result.selections["measurementMethod"].continued_by == ESTIMATED
    assert _served(result) == _expected(values)
    assert _served(result)[(KENYA, "2024")][1] is None
    # The trace's continuation starts where the served estimates do, after the overlap.
    minted = mint(PLACES, window=_WINDOW, facets_by_place=facets_by_place, values=values)
    assert minted.trace.to_dict()["continuations"] == _CONTINUED


_HOLDS_NOTHING = "Count_Person"
"""A group member's variable the fakes hold no facet of, so the group serves GDP's rows."""


_GBP_TO_2029 = {
    "CurrencyUSD": dict.fromkeys(range(2016, 2031), 1.0),
    "CurrencyGBP": dict.fromkeys(range(2016, 2030), 0.8),
}
"""Pound sterling rates through 2029: the window's 2030 has none, so a conversion of the
served rows to pounds refuses with a rate gap."""


@pytest.mark.parametrize(
    ("ceiling", "rates", "refusal"),
    [
        pytest.param(60, None, None, id="served-rows-fit"),
        pytest.param(59, None, OverCeilingError, id="served-rows-over"),
        # Converted to pounds, 2030 has no rate: the rows fit, so the rate gap refuses.
        pytest.param(60, _GBP_TO_2029, RateCoverageGapError, id="fit-with-a-rate-gap"),
        # Over the ceiling as well, so `over_ceiling` wins: it is counted before conversion.
        pytest.param(59, _GBP_TO_2029, OverCeilingError, id="over-with-a-rate-gap"),
    ],
)
@pytest.mark.parametrize("path", ["member", "group"])
def test_the_row_ceiling_counts_the_rows_left_after_the_reported_value_wins(
    path, ceiling, rates, refusal
):
    """With Kenya's overlapping 2024 estimate the read holds 61 rows and serves 60, four
    places by fifteen years, so a ceiling of 60 serves them and 59 refuses. The overlapping
    estimate is left out before the rows are counted, and the count precedes conversion, on
    a member's read and a group's."""
    facets_by_place, values = _kenya_overlap()
    bindings = (
        None
        if rates is None
        else {"unit": FacetBinding(state="request", request=UnitRequest("GBP", "current"))}
    )

    def run():
        if path == "member":
            return mint(
                PLACES,
                window=_WINDOW,
                facet_bindings=bindings,
                facets_by_place=facets_by_place,
                values=values,
                ceiling=ceiling,
                rates=rates,
            )
        return mint_group(
            PLACES,
            variables=(GDP, _HOLDS_NOTHING),
            holding=(GDP,),
            window=_WINDOW,
            facet_bindings=bindings,
            facets_by_place=facets_by_place,
            values=values,
            ceiling=ceiling,
            rates=rates,
        )

    if refusal is not None:
        with pytest.raises(refusal):
            run()
        return
    result = run()
    assert result.trace.to_dict()["continuations"] == _CONTINUED
    assert len(rows(result)) == 60


@pytest.mark.parametrize(
    ("reported", "method", "continued_by"),
    [
        # The estimates cover six of the window's years and the reported series two, yet the
        # two rank as one candidate led by the reported series, so its 2024 and 2025 values
        # serve and the estimates follow.
        pytest.param(True, None, ESTIMATED, id="reported-years-serve-then-estimates"),
        # With no reported series the estimates are elected alone.
        pytest.param(False, ESTIMATED, None, id="control-estimates-alone"),
    ],
)
def test_a_window_the_estimates_cover_more_of_keeps_the_reported_years(
    reported, method, continued_by
):
    window = RequestedWindow(kind="range", start_year=2024, end_year=2030)
    facets_by_place = {
        place: [f for f in facets if reported or f.measurement_method == ESTIMATED]
        for place, facets in recorded_facets().items()
    }

    result = preview(PLACES, window=window, facets_by_place=facets_by_place)

    assert result.selections["measurementMethod"].ref == method
    assert result.selections["measurementMethod"].continued_by == continued_by
    values = recorded_values()
    expected = {}
    for place in PLACES:
        for year in range(2024, 2031):
            key = str(year)
            series = (
                facet_id(place, unit=_USD, method=None)
                if reported and year <= _LAST_REPORTED[place]
                else facet_id(place, unit=_USD, method=ESTIMATED)
            )
            if key in values[(place, series)]:
                expected[(place, key)] = (
                    values[(place, series)][key],
                    None if series == facet_id(place, unit=_USD, method=None) else ESTIMATED,
                )
    assert _served(result) == expected


_UNLISTED = "provenance/ConstructedWithoutAReportedEntry"
"""A constructed provenance the reported list does not name."""


def _under(provenance: str) -> dict[str, list[Facet]]:
    """The recorded facets, every one moved under `provenance`."""
    return {
        place: [
            Facet(
                facet_id=candidate.facet_id,
                provenance_id=provenance,
                measurement_method=candidate.measurement_method,
                unit=candidate.unit,
                obs_count=candidate.obs_count,
                earliest_date=candidate.earliest_date,
                latest_date=candidate.latest_date,
            )
            for candidate in facets
        ]
        for place, facets in recorded_facets().items()
    }


_CONTINUED = [{"axis": "measurementMethod", "method": ESTIMATED, "periods": ["2025", "2030"]}]
"""The trace's continuation over the recorded data: the estimates supply 2025 at Kenya and
Zimbabwe and 2026 at France and Ecuador, through 2030."""


@pytest.mark.parametrize(
    ("window", "provenance", "estimated", "provisional", "continuations"),
    [
        # France 2026-2030, Kenya and Zimbabwe 2025-2030, Ecuador 2026-2030 are estimates,
        # and WEO's method-less rows count as reported. Each place's reported series and the
        # estimates continuing it count as one series reaching 2030.
        pytest.param(_WINDOW, WEO, 22, 0, _CONTINUED, id="listed-provenance-counts-its-estimates"),
        # The same rows under a provenance off the list: their method-less rows are unknown.
        pytest.param(
            _WINDOW, _UNLISTED, None, None, _CONTINUED, id="control-unlisted-provenance-is-unknown"
        ),
        # Only estimates in the window, four places from 2026 to 2030: every row classed, and
        # the estimates are elected alone, so nothing continues them.
        pytest.param(
            RequestedWindow(kind="range", start_year=2026, end_year=2030),
            WEO,
            20,
            0,
            None,
            id="control-only-estimated-rows-count",
        ),
    ],
)
def test_the_mints_figures_count_estimated_values_from_each_rows_method(
    window, provenance, estimated, provisional, continuations
):
    from materialise.statistics import reported_provenances

    assert WEO in reported_provenances()
    assert _UNLISTED not in reported_provenances()

    result = mint(PLACES, window=window, facets_by_place=_under(provenance))

    served_estimates = 22 if window == _WINDOW else 20
    assert sum(row["measurementMethod"] == ESTIMATED for row in rows(result)) == served_estimates
    figures = result.series_statistics.figures
    assert (figures.status.estimated, figures.status.provisional) == (estimated, provisional)
    assert result.trace.to_dict().get("continuations") == continuations
    # One series per place, each reaching 2030, however many methods serve it; the manifest's
    # `series_count` counts the same.
    assert result.series_statistics.series_count == len(PLACES)
    assert (figures.series, figures.latest_period, figures.series_reaching_latest) == (
        len(PLACES),
        "2030",
        len(PLACES),
    )
    assert [(each.source, each.series) for each in figures.by_source_unit] == [
        (provenance, len(PLACES))
    ]


def test_a_joined_candidates_choice_reports_the_figures_the_ranking_compared():
    """From 2024 to 2030 the US dollar candidate is the reported series joined to the
    estimates continuing it, so the unit choice the trace records gives it all seven window
    years and the observations of both, where its reported series alone covers two."""
    window = RequestedWindow(kind="range", start_year=2024, end_year=2030)

    choices = mint(PLACES, window=window).trace.to_dict()["choices"]

    usd = [
        candidate
        for choice in choices
        if choice["axis"] == "unit"
        for candidate in choice["candidates"]
        if candidate["ref"] == _USD
    ]
    joined_observations = sum(
        f.obs_count for facets in recorded_facets().values() for f in facets if f.unit == _USD
    )
    assert usd == [
        {"ref": _USD, "places": len(PLACES), "years": 7, "observations": joined_observations}
    ]


@pytest.mark.parametrize("call", [preview, mint])
@pytest.mark.parametrize("national_currency", [False, True], ids=["unbound", "national-currency"])
@pytest.mark.parametrize(
    ("end_year", "estimated"),
    [
        # France's and Ecuador's reported series reach 2025, Kenya's and Zimbabwe's end in
        # 2024, so the estimates serve those two places' 2025 and nothing else.
        pytest.param(
            2025,
            {(KENYA, "2025"), (ZIMBABWE, "2025")},
            id="estimates-continue-the-series-ending-before-the-windows-end",
        ),
        # Every place's reported series reaches 2024, so no estimate serves.
        pytest.param(2024, set(), id="control-every-series-reaches-the-windows-end"),
    ],
)
def test_each_series_continues_past_its_own_last_reported_year(
    call, national_currency, end_year, estimated
):
    from .test_accept_national_currency import _national_currency

    window = RequestedWindow(kind="range", start_year=2020, end_year=end_year)

    result = call(
        PLACES, window=window, facet_bindings=_national_currency() if national_currency else None
    )

    methods = {
        (str(row["observationAbout"]), str(row["time_period"])): row["measurementMethod"]
        for row in rows(result)
    }
    assert methods == {
        (place, str(year)): ESTIMATED if (place, str(year)) in estimated else None
        for place in PLACES
        for year in range(2020, end_year + 1)
    }


@pytest.mark.parametrize("group", [False, True], ids=["member", "group"])
def test_a_continuation_that_serves_no_row_is_left_out_of_the_series_read(group: bool):
    """Kenya's and Zimbabwe's estimates hold no value in 2025, so the estimated method
    continues their series and serves no row. The trace names no continuation, and the
    series read are those of the facets the trace names as dispatched."""
    values = recorded_values()
    for place in (KENYA, ZIMBABWE):
        estimated = facet_id(place, unit=_USD, method=ESTIMATED)
        values[(place, estimated)] = {
            year: value for year, value in values[(place, estimated)].items() if year != "2025"
        }

    window = RequestedWindow(kind="range", start_year=2020, end_year=2025)

    minted = (
        mint_group(
            PLACES, variables=(GDP, _HOLDS_NOTHING), holding=(GDP,), window=window, values=values
        )
        if group
        else mint(PLACES, window=window, values=values)
    )

    trace = minted.trace.to_dict()
    read = minted.read_series[f"dcid:{GDP}"]
    assert "continuations" not in trace
    assert set(read.facet_ids) == {
        cast(str, facet["dispatch_facet_id"])
        for facet in cast(list[dict[str, object]], trace["facets"])
    }
