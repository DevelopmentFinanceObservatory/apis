"""A mint whose window opts in to nothing holds no row of a source's partial period and no
projected row, so every computation over the frame reads complete, actual years
(`materialise.held_back`, `materialise.columnar.held_back_mask`), and the record of an opted-in
frame names what it holds (`datasets_api.manifest.incomplete_after`).

Two fakes stand for the sources. A CRS-like source, marked partial for 2025 as the dev graph's
CRS node marks it, holds ninety recipients from 2015, some ending in 2024 and some reaching
2025. An IDS-like source holds three debtors with reported values to 2025 and `Estimated`
values from 2026 to 2030, a projection pair `method_status.json` declares.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import replace
from functools import partial
from typing import Any

import pytest

from datasets_api.manifest import held_back_coverage, incomplete_after
from dfo_analysis_engine.anchor import latest_period
from dfo_analysis_engine.calls import admit
from dfo_analysis_engine.engine.aggregate import run_aggregate
from dfo_analysis_engine.engine.growth import run_growth
from dfo_analysis_engine.op_spec import RootPopulation
from materialise import held_back as held_back_module
from materialise.facets import FacetBinding, NationalCurrency
from materialise.graph import Facet
from materialise.pipeline import MaterialisationResult
from materialise.window import RequestedWindow, election_window
from origin_catalogue import CatalogueNotBound, bound_catalogue
from origin_catalogue import context as catalogue_context
from tests.origin_catalogue.fake_catalogue import dev_provenance_nodes, documenting

from . import _weo_recorded
from ._weo_recorded import Values, mint, mint_group, preview, rows
from .conftest import only_chunk

_CRS = "provenance/OECDCreditorReportingSystem"
_IDS = "provenance/WorldBankInternationalDebtStatistics"
_OTHER = "provenance/SomeEstimatingSource"
_MONTHLY = "provenance/SomeMonthlySource"
_USD = "USD"
_ESTIMATED = "Estimated"
_SPEC = "spec_" + "0" * 32


@pytest.fixture(autouse=True)
def _sdmx_serving_the_periods_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    """SDMX serves each series' newest row to a `LATEST` read and only the years a read of
    a year asks for, so a `latest` read whose newest rows are held back must read again."""
    monkeypatch.setattr(_weo_recorded, "sdmx", partial(_weo_recorded.sdmx, honour_periods=True))


@pytest.fixture(autouse=True)
def _marked_catalogue() -> Iterator[None]:
    """CRS marks 2025 partial, as its dev graph node does, and the monthly source June
    2025; IDS and the other source mark nothing."""
    unmarked = {"lastDataRefreshDate": ("2026-10-29",)}
    monthly = {**unmarked, "partialObservationDate": ("2025-06",)}
    with bound_catalogue(
        documenting(
            {
                _CRS: dev_provenance_nodes()[_CRS],
                _IDS: unmarked,
                _OTHER: unmarked,
                _MONTHLY: monthly,
            }
        )
    ):
        yield


def _facet(place: str, *, provenance: str, method: str | None, years: range) -> Facet:
    return Facet(
        facet_id=f"{provenance.rsplit('/', 1)[-1]}-{place}-{method or 'reported'}",
        provenance_id=provenance,
        measurement_method=method,
        unit=_USD,
        obs_count=len(years),
        earliest_date=str(years.start),
        latest_date=str(years.stop - 1),
    )


type _Source = tuple[list[str], dict[str, list[Facet]], Values]


def _source(series: dict[str, list[tuple[str, str | None, range]]]) -> _Source:
    """The places, facets and values of `series`: each place's `(provenance, method, years)`
    facets, each year's value its year."""
    facets = {
        place: [_facet(place, provenance=p, method=m, years=years) for p, m, years in held]
        for place, held in series.items()
    }
    values = {
        (place, facet.facet_id): {str(year): float(year) for year in years}
        for place, held in series.items()
        for facet, (_, _, years) in zip(facets[place], held, strict=True)
        if facet.facet_id is not None
    }
    return list(series), facets, values


def _crs(*, ending_2024: int, reaching_2025: int) -> _Source:
    return _source(
        {
            f"country/R{i:02d}": [(_CRS, None, range(2015, 2025 if i < ending_2024 else 2026))]
            for i in range(ending_2024 + reaching_2025)
        }
    )


_RAGGED = pytest.mark.parametrize(
    "crs",
    [
        pytest.param(_crs(ending_2024=80, reaching_2025=10), id="ten-of-ninety-reach-2025"),
        pytest.param(_crs(ending_2024=10, reaching_2025=80), id="eighty-of-ninety-reach-2025"),
    ],
)


def _mint(source: _Source, window: RequestedWindow, **bindings: Any) -> MaterialisationResult:
    places, facets, values = source
    return mint(
        places,
        window=window,
        facets_by_place=facets,
        values=values,
        currencies={},
        facet_bindings=bindings or None,
    )


def _years(result: Any) -> set[str]:
    return {str(row["time_period"]) for row in rows(result)}


_OPEN = RequestedWindow(kind="range", start_year=2015)


@_RAGGED
def test_an_open_range_holds_no_row_of_the_partial_year_and_records_nothing_partial(
    crs: _Source,
) -> None:
    result = _mint(crs, _OPEN)

    assert max(_years(result)) == "2024"
    assert {row["observationAbout"] for row in rows(result) if row["time_period"] == "2024"} == set(
        crs[0]
    )
    judged = incomplete_after(result)
    assert judged is not None
    assert judged.entries == []


def test_a_range_opting_in_to_partial_periods_holds_them_and_records_one_partial_entry() -> None:
    crs = _crs(ending_2024=80, reaching_2025=10)

    result = _mint(
        crs, RequestedWindow(kind="range", start_year=2015, include=frozenset({"partial"}))
    )

    assert sum(row["time_period"] == "2025" for row in rows(result)) == 10
    judged = incomplete_after(result)
    assert judged is not None
    assert [(entry["status"], entry["partial_period"]) for entry in judged.entries] == [
        ("partial", "2025")
    ]


def _latest_mint(source: _Source) -> Any:
    places, facets, values = source
    return mint(
        places,
        window=RequestedWindow(kind="latest"),
        facets_by_place=facets,
        values=values,
        currencies={},
    )


def _latest_group(source: _Source) -> Any:
    places, facets, values = source
    return mint_group(
        places,
        variables=["GdpLike"],
        holding=["GdpLike"],
        window=RequestedWindow(kind="latest"),
        facets_by_place=facets,
        values=values,
    )


def _latest_preview(source: _Source) -> Any:
    places, facets, values = source
    return preview(
        places,
        window=RequestedWindow(kind="latest"),
        facets_by_place=facets,
        values=values,
        currencies={},
    )


@_RAGGED
@pytest.mark.parametrize(
    "read",
    [
        pytest.param(_latest_mint, id="member"),
        pytest.param(_latest_group, id="group"),
        pytest.param(_latest_preview, id="preview"),
    ],
)
def test_latest_serves_the_newest_complete_year_from_every_series(
    crs: _Source, read: Callable[[_Source], Any]
) -> None:
    served = rows(read(crs))

    assert {row["time_period"] for row in served} == {"2024"}
    assert {row["observationAbout"] for row in served} == set(crs[0])


@_RAGGED
def test_a_computation_over_the_default_frame_reads_no_partial_year(crs: _Source) -> None:
    """The rolling anchor, a sum across series and a growth to the anchor's year each read
    the minted frame as the analysis engine reads it."""
    result = _mint(crs, _OPEN)
    frame = only_chunk(result)
    root = admit(frame, content_version=result.content_version)

    anchor = latest_period([RootPopulation(spec_id=_SPEC)], [(_SPEC, root)])
    summed = run_aggregate(frame, group_by=("time_period",), aggregation="sum")
    growth = run_growth(
        frame, window_start="2015", window_end=anchor.period, entity_by=("observationAbout",)
    )

    assert anchor.period == "2024"
    assert summed.collapsed_periods[-1] == "2024"
    assert len(growth.entity_growths) == len(crs[0])
    assert growth.observed_periods[-1] == "2024"


_IDS_DEBTORS = _source(
    {
        f"country/D{i}": [
            (_IDS, None, range(2015, 2026)),
            (_IDS, _ESTIMATED, range(2026, 2031)),
        ]
        for i in range(3)
    }
)
_TO_2030 = RequestedWindow(kind="range", start_year=2015, end_year=2030)
_BOUND_ESTIMATED = {"measurementMethod": FacetBinding(state="bound", ref=_ESTIMATED)}


@pytest.mark.parametrize(
    "window",
    [
        pytest.param(election_window(_OPEN, clock_year=2026), id="open-range"),
        pytest.param(_TO_2030, id="closed-range-past-2025"),
    ],
)
def test_a_range_past_the_last_reported_year_holds_no_projected_row(
    window: RequestedWindow,
) -> None:
    result = _mint(_IDS_DEBTORS, window)

    assert max(_years(result)) == "2025"
    assert {row["measurementMethod"] for row in rows(result)} == {None}


def _runs(
    result: MaterialisationResult, places: list[str], window: RequestedWindow
) -> list[tuple[object, ...]]:
    """The `held_back` runs the mint states, without their shares."""
    request = _weo_recorded._request(places, window=window, facet_bindings=None)
    return [
        (held["kind"], held["served"], held["source"], held["start_year"], held["end_year"])
        for held in held_back_coverage(result, requests=[request])
    ]


def test_a_request_binding_the_projection_method_holds_no_row_without_the_opt_in() -> None:
    """The read dispatches only projection facets, so their years are the projected run it
    states held back, and the empty frame says why."""
    result = _mint(_IDS_DEBTORS, _TO_2030, **_BOUND_ESTIMATED)

    assert rows(result) == []
    assert _runs(result, _IDS_DEBTORS[0], _TO_2030) == [("projected", False, _IDS, 2026, 2030)]


@pytest.mark.parametrize(
    "bindings", [pytest.param({}, id="continued"), pytest.param(_BOUND_ESTIMATED, id="bound")]
)
def test_a_range_opting_in_to_projections_holds_them_and_records_a_projected_entry(
    bindings: dict[str, FacetBinding],
) -> None:
    window = RequestedWindow(
        kind="range", start_year=2015, end_year=2030, include=frozenset({"projected"})
    )

    result = _mint(_IDS_DEBTORS, window, **bindings)

    projected = {row["time_period"] for row in rows(result) if row["measurementMethod"]}
    assert projected == {str(year) for year in range(2026, 2031)}
    judged = incomplete_after(result)
    assert judged is not None
    assert [
        (entry["status"], entry["source"], entry["partial_period"]) for entry in judged.entries
    ] == [("projected", _IDS, "2026")]
    assert _runs(result, _IDS_DEBTORS[0], window) == [("projected", True, _IDS, 2026, 2030)]


def test_an_undeclared_estimated_continuation_is_served() -> None:
    source = _source(
        {
            "country/D0": [
                (_OTHER, None, range(2015, 2026)),
                (_OTHER, _ESTIMATED, range(2026, 2031)),
            ]
        }
    )

    result = _mint(source, _TO_2030)

    assert max(_years(result)) == "2030"
    judged = incomplete_after(result)
    assert judged is not None
    assert judged.entries == []


def _months(first: str, last: str) -> list[str]:
    year, month = int(first[:4]), int(first[5:])
    months = []
    while (period := f"{year}-{month:02d}") <= last:
        months.append(period)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


@pytest.mark.parametrize(
    "read",
    [
        pytest.param(_latest_mint, id="member"),
        pytest.param(_latest_group, id="group"),
        pytest.param(_latest_preview, id="preview"),
    ],
)
def test_latest_serves_a_monthly_series_the_newest_month_before_its_partial_month(
    read: Callable[[_Source], Any],
) -> None:
    """The source marks June 2025 partial, so January to May 2025 are complete, and a series
    ending in 2024 holds none of them."""
    ends = {"country/M0": "2025-09", "country/M1": "2025-09", "country/M2": "2024-11"}
    facets = {
        place: [
            Facet(
                facet_id="monthly",
                provenance_id=_MONTHLY,
                measurement_method=None,
                unit=_USD,
                obs_count=len(_months("2023-01", end)),
                earliest_date="2023-01",
                latest_date=end,
            )
        ]
        for place, end in ends.items()
    }
    values = {
        (place, "monthly"): dict.fromkeys(_months("2023-01", end), 1.0)
        for place, end in ends.items()
    }

    served = rows(read((list(ends), facets, values)))

    assert {(row["observationAbout"], row["time_period"]) for row in served} == {
        ("country/M0", "2025-05"),
        ("country/M1", "2025-05"),
    }


def _in_two_currencies(source: _Source) -> tuple[_Source, dict[str, list[str]]]:
    """`source` with every other place's facets in euros and the rest in US dollars, and
    each place's national currency, the one its facets state."""
    places, facets, values = source
    units = {place: "CurrencyEUR" if i % 2 else "CurrencyUSD" for i, place in enumerate(places)}
    facets = {place: [replace(f, unit=units[place]) for f in facets[place]] for place in places}
    return (places, facets, values), {place: [unit] for place, unit in units.items()}


def _share(result: MaterialisationResult, places: list[str], window: RequestedWindow) -> object:
    """The reported share of the partial run the mint's `held_back` states."""
    request = _weo_recorded._request(places, window=window, facet_bindings=None)
    (run,) = [
        held for held in held_back_coverage(result, requests=[request]) if held["kind"] == "partial"
    ]
    return run["reported_share"]


_RANGE = RequestedWindow(kind="range", start_year=2016, end_year=2018)
_LATEST = RequestedWindow(kind="latest")
_TEN_OF_NINETY = _crs(ending_2024=80, reaching_2025=10)
_TWO_CURRENCIES, _NATIONAL_CURRENCIES = _in_two_currencies(_TEN_OF_NINETY)
_VALUE_SHARE = {"basis": "value", "share": 10 * 2025 / (90 * 2024), "against": 2024}
"""Each value is its year, so 2025's ten series' values over 2024's ninety's."""
_SERIES_SHARE = {"basis": "series", "share": 10 / 90, "against": 2024}


@pytest.mark.parametrize(
    ("entry", "source", "currencies", "window", "share"),
    [
        pytest.param("member", _TEN_OF_NINETY, {}, _RANGE, _VALUE_SHARE, id="member-range"),
        pytest.param("group", _TEN_OF_NINETY, {}, _RANGE, _VALUE_SHARE, id="group-range"),
        # A group binds no national currency, so a member alone reads two currencies.
        pytest.param(
            "member",
            _TWO_CURRENCIES,
            _NATIONAL_CURRENCIES,
            _RANGE,
            _SERIES_SHARE,
            id="member-range-two-currencies",
        ),
        pytest.param("member", _TEN_OF_NINETY, {}, _LATEST, _VALUE_SHARE, id="member-latest"),
        pytest.param("group", _TEN_OF_NINETY, {}, _LATEST, _VALUE_SHARE, id="group-latest"),
        pytest.param(
            "member",
            _TWO_CURRENCIES,
            _NATIONAL_CURRENCIES,
            _LATEST,
            _SERIES_SHARE,
            id="member-latest-two-currencies",
        ),
    ],
)
def test_a_held_back_partial_year_states_the_share_its_read_counted(
    entry: str,
    source: _Source,
    currencies: dict[str, list[str]],
    window: RequestedWindow,
    share: dict[str, object],
) -> None:
    """The window holds 2025 back, so the share is counted over the rows the read held
    before its cuts: 2025's held-back rows and the 2024 rows a range leaves out or a `latest`
    read reads again, on the value basis on either window. Rows in two currencies count by
    series, since no sum adds them."""
    places, facets, values = source
    if entry == "group":
        result = mint_group(
            places,
            variables=["GdpLike"],
            holding=["GdpLike"],
            window=window,
            facets_by_place=facets,
            values=values,
        )
    else:
        national = NationalCurrency(role="observationAbout")
        result = mint(
            places,
            window=window,
            facets_by_place=facets,
            values=values,
            currencies=currencies,
            facet_bindings=(
                {"unit": FacetBinding(state="national_currency", national_currency=national)}
                if currencies
                else None
            ),
        )

    assert "2025" not in _years(result)
    assert _share(result, places, window) == pytest.approx(share)


def test_a_projection_in_another_national_currency_states_its_run() -> None:
    """Under a national-currency binding each debtor reads its own currency. Only the
    euro debtor's series ends in 2022 and is projected from 2023, so the run starts there,
    whichever debtor's facet leads the read."""
    places = ["country/D0", "country/D1"]
    facets = {
        "country/D0": [_facet("country/D0", provenance=_IDS, method=None, years=range(2015, 2026))],
        "country/D1": [
            replace(f, unit="CurrencyEUR")
            for f in (
                _facet("country/D1", provenance=_IDS, method=None, years=range(2015, 2023)),
                _facet("country/D1", provenance=_IDS, method=_ESTIMATED, years=range(2023, 2031)),
            )
        ],
    }
    facets["country/D0"] = [replace(f, unit="CurrencyUSD") for f in facets["country/D0"]]
    values = {
        (place, f.facet_id): {str(y): float(y) for y in range(2015, 2031)}
        for place in places
        for f in facets[place]
        if f.facet_id is not None
    }
    national = NationalCurrency(role="observationAbout")

    result = mint(
        places,
        window=_TO_2030,
        facets_by_place=facets,
        values=values,
        currencies={"country/D0": ["CurrencyUSD"], "country/D1": ["CurrencyEUR"]},
        facet_bindings={
            "unit": FacetBinding(state="national_currency", national_currency=national)
        },
    )

    assert _runs(result, places, _TO_2030) == [("projected", False, _IDS, 2023, 2030)]


def test_a_mint_with_no_catalogue_bound_refuses_even_when_it_opts_in_to_every_year() -> None:
    """The cut of a window opting in to both kinds needs no catalogue, but the runs it states
    read the partial marks, so it refuses as Search does rather than state none."""
    window = RequestedWindow(
        kind="range", start_year=2015, end_year=2030, include=frozenset({"partial", "projected"})
    )
    token = catalogue_context._BOUND.set(None)
    try:
        with pytest.raises(CatalogueNotBound):
            _mint(_IDS_DEBTORS, window)
    finally:
        catalogue_context._BOUND.reset(token)


def test_rows_of_a_source_the_catalogue_does_not_document_are_served_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No mark can be read for the source, so its rows are served unjudged, and the cut says
    so once."""
    source = "provenance/SomeUndocumentedSource"
    held_back_module._warn_undocumented.cache_clear()

    with caplog.at_level(logging.WARNING):
        result = _mint(_source({"country/U0": [(source, None, range(2015, 2026))]}), _OPEN)

    assert max(_years(result)) == "2025"
    assert [r.args for r in caplog.records if r.name == held_back_module.__name__] == [(source,)]


@pytest.mark.unbound_release
def test_a_judgement_with_no_release_bound_refuses_to_read_a_source_as_a_plain_table() -> None:
    """A source's partial marks are its dataset members', which only the bound release's code
    index names, so a judgement with none bound raises, as every member-map reader does."""
    from graph_release.context import ReleaseNotBound

    with (
        bound_catalogue(documenting({_CRS: dev_provenance_nodes()[_CRS]})),
        pytest.raises(ReleaseNotBound),
    ):
        held_back_module.judge(
            source=_CRS, method=None, period="2025", catalogue=catalogue_context.current_catalogue()
        )
