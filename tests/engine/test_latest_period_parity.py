"""A `latest` answer and a Datasets read of the same StatVar serve one period, the newest any
of its series holds, from the same entities (`materialise.window.latest_period`), on a
role-bearing read with an open role and on a place-keyed read alike.

Each world holds one StatVar at Aland and Bland, whose series end in different years or
whose coverage index disagrees with SDMX. Search answers through the first door of the
answer-invariants world (`_answer_invariants_world`), and Datasets previews the same StatVar
through `materialise_preview`, which reduces a `latest` window as a mint does.
"""

from __future__ import annotations

import logging
from typing import cast

import pytest

from materialise import MaterialisationRequest, RequestedWindow, materialise_preview
from materialise.facets import FacetBinding, UnitRequest
from materialise.rate_tables import RateTableCache
from materialise.recipe_bodies import RECIPE_BODIES
from search.engine.region.codes import OBSERVATIONS_PROBE_FAILED
from search.engine.region.observations.fetch import _at_latest_periods, _SeriesRead
from search.models import TimeWindow, Warning
from tests.engine import _answer_invariants_world as world_module
from tests.engine._answer_invariants_world import (
    ALAND,
    BLAND,
    CREDITOR,
    DATES,
    EUR,
    USD,
    Row,
    World,
)
from tests.materialise._rate_table_support import recipe_table_bodies

_ALTCRED = "country/ALT"
_TO_2020, _TO_2022 = tuple(range(2010, 2021)), tuple(range(2012, 2023))
_TO_2023, _TO_2024 = tuple(range(2012, 2024)), tuple(range(2012, 2025))
_INDEX_DISAGREES = "the coverage index and the observations read disagree"


def _world(route: str, holdings: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]) -> World:
    """One StatVar on `route`, each `(place, indexed years, SDMX years)` in `holdings` a
    series the coverage index dates by the first years and SDMX returns the second."""
    stat_var = f"TESTFIXTURE/LatestParity/{'Amount' if route == 'standard' else 'Flow'}"
    return World(
        route=route,
        rows=tuple(
            Row(stat_var, "f-ids", place, indexed, sdmx_years=read)
            for place, indexed, read in holdings
        ),
        stat_vars={stat_var: ("latestParityAmount", None)},
    )


def _search_served(
    world: World,
) -> tuple[set[tuple[str, str]], TimeWindow | None, tuple[Warning, ...]]:
    """The `(place, period)` points the first answer's preview serves under `latest`, the
    window its coverage states and the answer's warnings."""
    with world_module.door_loop():
        answer = world.first([ALAND, BLAND], DATES["latest"], preview=True, unit_text=None)
    (spec,) = answer.specs
    observations = spec.resolved.observations
    assert observations is not None and observations.omitted is None
    place_role = "destination" if world.role else "observationAbout"
    served = {
        (next(e.ref for e in s.entities if e.role == place_role), p.date)
        for s in observations.series
        for p in s.points
    }
    assert answer.result is not None
    return served, spec.resolved.coverage.window, answer.result.warnings


def _datasets_served(
    world: World, stat_var: str | None = None, *, unit_text: str | None = None
) -> set[tuple[str, str]]:
    """The `(place, period)` rows a Datasets read of `stat_var`, the world's one StatVar by
    default, serves under `latest`. A place-keyed read names the world's places, and a role
    read takes every recipient."""
    if stat_var is None:
        (stat_var,) = world.stat_vars
    place_role = "debtor" if world.role else "observationAbout"
    preview = materialise_preview(
        MaterialisationRequest(
            stat_var=f"dcid:{stat_var}",
            role_bindings={} if world.role else {place_role: [ALAND, BLAND]},
            declared_roles=world.declared_roles,
            facet_bindings=(
                {
                    "unit": FacetBinding(
                        state="request", request=UnitRequest(currency=unit_text, prices="current")
                    )
                }
                if unit_text is not None
                else {}
            ),
            window=RequestedWindow(kind="latest"),
        ),
        sdmx=world.materialise_sdmx,
        graph=world.materialise_graph,
        limit=10_000,
        coverage=world.index,
        rate_cache=RateTableCache(),
    )
    columns = preview.frame.columns
    place, period = columns.index(place_role), columns.index("time_period")
    return {(cast(str, row[place]), cast(str, row[period])) for row in preview.frame.rows}


@pytest.mark.parametrize(
    "route",
    [
        pytest.param("discovered_pair", id="open-role"),
        pytest.param("standard", id="place-keyed"),
    ],
)
def test_latest_uses_the_newest_period_sdmx_returns_when_coverage_is_ahead(
    route: str, search_settings
) -> None:
    """When coverage ends in 2023 but both SDMX series end in 2022, Search serves the
    same actual latest rows Datasets does rather than dropping every returned point."""
    search_settings(obs_preview_max_pairs=120)
    world = _world(
        route,
        ((ALAND, _TO_2023, _TO_2022), (BLAND, _TO_2023, _TO_2022)),
    )

    served, _window, _warnings = _search_served(world)

    assert served == _datasets_served(world) == {(ALAND, "2022"), (BLAND, "2022")}


@pytest.mark.parametrize(
    "route",
    [
        pytest.param("discovered_pair", id="open-role"),
        pytest.param("standard", id="place-keyed"),
    ],
)
def test_latest_recovers_the_actual_winner_excluded_by_coverage(
    route: str, search_settings
) -> None:
    """Aland's indexed 2023 point does not exist in SDMX, while Bland's indexed 2022 point
    is real. Search must prepare both held series and serve Bland's actual latest period."""
    search_settings(obs_preview_max_pairs=120)
    world = _world(
        route,
        ((ALAND, _TO_2023, _TO_2020), (BLAND, _TO_2022, _TO_2022)),
    )

    served, _window, _warnings = _search_served(world)

    assert served == _datasets_served(world) == {(BLAND, "2022")}


@pytest.mark.parametrize(
    ("holdings", "period", "short", "outside"),
    [
        pytest.param(
            ((ALAND, _TO_2020, _TO_2020), (BLAND, _TO_2023, _TO_2023)),
            "2023",
            0,
            False,
            id="ends-differ",
        ),
        pytest.param(
            ((ALAND, _TO_2023, _TO_2022), (BLAND, _TO_2023, _TO_2023)),
            "2023",
            1,
            False,
            id="index-ahead",
        ),
        pytest.param(
            ((ALAND, _TO_2023, _TO_2023), (BLAND, _TO_2023, _TO_2024)),
            "2024",
            1,
            True,
            id="index-lags",
        ),
    ],
)
@pytest.mark.parametrize(
    "route",
    [
        pytest.param("discovered_pair", id="open-role"),
        pytest.param("standard", id="place-keyed"),
    ],
)
def test_search_and_datasets_serve_one_latest_period_from_the_same_entities(
    route: str,
    holdings: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...],
    period: str,
    short: int,
    outside: bool,
    search_settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Both surfaces serve the StatVar's newest period from Bland alone, and Search's
    coverage names the year the index dates it at.

    Where Aland's series ends in 2020 the plan never serves it. Where the index dates Aland
    at 2023 and SDMX ends it in 2022, Search reads it and drops it, and the partial warning
    and the log name it. Where SDMX holds 2024 for Bland past the index's 2023, Search reads
    from 2023 onward and serves 2024 as the mint does, drops Aland's 2023, and logs the
    point outside its coverage window."""
    search_settings(obs_preview_max_pairs=120)
    world = _world(route, holdings)

    with caplog.at_level(logging.WARNING):
        served, window, warnings = _search_served(world)

    assert served == _datasets_served(world) == {(BLAND, period)}
    assert window == TimeWindow(start_year=2023, end_year=2023)
    partial = [
        w.message
        for w in warnings
        if w.code == OBSERVATIONS_PROBE_FAILED and "could not be served" in w.message
    ]
    assert partial == (
        [f"1 of 2 requested series are included; {short} could not be served."] if short else []
    )
    dropped = [r for r in caplog.records if "planned series held no point" in r.getMessage()]
    assert [r.args for r in dropped] == ([(short, [_series_key(world, ALAND)])] if short else [])
    outside_logged = any(
        "outside its coverage window" in r.getMessage() and _INDEX_DISAGREES in r.getMessage()
        for r in caplog.records
    )
    assert outside_logged is outside


def _series_key(world: World, place: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    """The planned series key Search's per-series preview gives `place`'s series, its roles
    sorted by name as a mint keys them."""
    (stat_var,) = world.stat_vars
    if world.role:
        return stat_var, (("creditor", CREDITOR), ("debtor", place))
    return stat_var, (("observationAbout", place),)


def test_a_rate_gap_in_a_dropped_series_year_refuses_neither_surface(
    search_settings, monkeypatch
) -> None:
    """An ask for flows to Bland reads one preview with the creditor open. Credland's series
    ends in 2023 and Altcred's in 2020, a year the rate table holds no euro rate for. Both
    surfaces serve Credland's 2023 in euros. A mint reduces before it converts, and Search
    reads the StatVar at its latest year alone, so Altcred's 2020 row is never converted."""
    search_settings(obs_preview_max_pairs=120)
    years = range(1990, 2031)
    rates = {USD: dict.fromkeys(years, 1.0), EUR: {y: 0.9 for y in years if y != 2020}}
    recipe = RECIPE_BODIES["ConversionRecipeDAC"]
    monkeypatch.setattr(
        world_module,
        "_RATE_BODY",
        recipe_table_bodies(recipe.recipe_id, rates=rates)[recipe.exchange_rate_variable],
    )
    stat_var = "TESTFIXTURE/LatestParity/Flow"
    world = World(
        route="named_pair",
        rows=(
            Row(stat_var, "f-ids", BLAND, tuple(range(2012, 2024))),
            Row(stat_var, "f-ids", BLAND, tuple(range(2010, 2021)), creditor=_ALTCRED),
        ),
        stat_vars={stat_var: ("latestParityAmount", None)},
    )

    with world_module.door_loop():
        answer = world.first([BLAND], DATES["latest"], preview=True, unit_text="EUR")
    (spec,) = answer.specs
    observations = spec.resolved.observations
    assert observations is not None and observations.omitted is None
    served = {
        (next(e.ref for e in s.entities if e.role == "origin"), p.date)
        for s in observations.series
        for p in s.points
    }
    preview = materialise_preview(
        MaterialisationRequest(
            stat_var=f"dcid:{stat_var}",
            role_bindings={"debtor": [BLAND]},
            declared_roles=world.declared_roles,
            facet_bindings={
                "unit": FacetBinding(
                    state="request", request=UnitRequest(currency="EUR", prices="current")
                )
            },
            window=RequestedWindow(kind="latest"),
        ),
        sdmx=world.materialise_sdmx,
        graph=world.materialise_graph,
        limit=10_000,
        coverage=world.index,
        rate_cache=RateTableCache(),
    )
    columns = preview.frame.columns
    creditor, period = columns.index("creditor"), columns.index("time_period")
    datasets = {(cast(str, row[creditor]), cast(str, row[period])) for row in preview.frame.rows}

    assert served == datasets == {(CREDITOR, "2023")}
    assert spec.resolved.coverage.window == TimeWindow(start_year=2023, end_year=2023)


def test_a_converted_latest_read_ignores_a_rate_gap_before_its_actual_latest_period(
    search_settings, monkeypatch
) -> None:
    """Coverage ends in 2023 while SDMX has Aland through 2024 and Bland through 2025.
    The absent 2024 euro rate cannot reject Bland's 2025 latest row: Search reduces the
    observations before conversion, as Datasets does."""
    search_settings(obs_preview_max_pairs=120)
    years = range(1990, 2031)
    rates = {USD: dict.fromkeys(years, 1.0), EUR: {y: 0.9 for y in years if y != 2024}}
    recipe = RECIPE_BODIES["ConversionRecipeDAC"]
    monkeypatch.setattr(
        world_module,
        "_RATE_BODY",
        recipe_table_bodies(recipe.recipe_id, rates=rates)[recipe.exchange_rate_variable],
    )
    stat_var = "TESTFIXTURE/LatestParity/ConvertedFlow"
    world = World(
        route="discovered_pair",
        rows=(
            Row(stat_var, "f-ids", ALAND, (2023,), sdmx_years=(2023, 2024)),
            Row(stat_var, "f-ids", BLAND, (2023,), sdmx_years=(2023, 2024, 2025)),
        ),
        stat_vars={stat_var: ("latestParityAmount", None)},
    )

    with world_module.door_loop():
        answer = world.first([ALAND, BLAND], DATES["latest"], preview=True, unit_text="EUR")

    assert answer.refusal is None
    assert answer.error is None
    assert answer.result is not None
    (spec,) = answer.specs
    observations = spec.resolved.observations
    assert observations is not None and observations.omitted is None
    served = {
        (next(e.ref for e in s.entities if e.role == "destination"), p.date)
        for s in observations.series
        for p in s.points
    }

    assert served == _datasets_served(world, unit_text="EUR") == {(BLAND, "2025")}
    assert all(series.conversion is not None for series in observations.series)
    assert [point.value for series in observations.series for point in series.points] == [
        pytest.approx(0.9)
    ]
    assert all(warning.code != "rate_missing" for warning in answer.result.warnings)


@pytest.mark.parametrize(
    "aland_indexed_to",
    [pytest.param(2020, id="index-agrees"), pytest.param(2023, id="index-ahead")],
)
def test_a_bulk_read_serves_each_statvar_its_own_latest_period(
    aland_indexed_to: int, search_settings, caplog: pytest.LogCaptureFixture
) -> None:
    """Two StatVars read in bulk under one elected facet, the facet a mint of each elects.
    Flow A's series end in 2020 and 2023, flow B's in 2018 and 2016. Each StatVar serves its
    own newest period from the series holding it, A 2023 at Bland and B 2018 at Aland, as a
    Datasets read of each does, and coverage spans the two years. Where the index dates flow
    A at Aland to 2023, the bulk read dispatches that pair, finds no 2023 point, drops it and
    logs it."""
    search_settings(obs_preview_max_pairs=1)
    flow_a, flow_b = "TESTFIXTURE/LatestParity/FlowA", "TESTFIXTURE/LatestParity/FlowB"
    world = World(
        route="discovered_bulk",
        rows=(
            Row(
                flow_a,
                "f-ids",
                ALAND,
                tuple(range(2010, aland_indexed_to + 1)),
                sdmx_years=tuple(range(2010, 2021)),
            ),
            Row(flow_a, "f-ids", BLAND, tuple(range(2012, 2024))),
            Row(flow_b, "f-ids", ALAND, tuple(range(2010, 2019))),
            Row(flow_b, "f-ids", BLAND, tuple(range(2010, 2017))),
        ),
        stat_vars={flow_a: ("latestParityAmount", None), flow_b: ("latestParityAmount", None)},
    )

    with caplog.at_level(logging.WARNING), world_module.door_loop():
        answer = world.first([ALAND, BLAND], DATES["latest"], preview=True, unit_text=None)
    (spec,) = answer.specs
    observations = spec.resolved.observations
    assert observations is not None and observations.omitted is None
    assert answer.recorder.counts()["sdmx.role_series"] == 1
    dropped = [r for r in caplog.records if "the bulk role read disagree" in r.getMessage()]
    assert [r.args for r in dropped] == (
        [(1, [(flow_a, (("creditor", CREDITOR), ("debtor", ALAND)))])]
        if aland_indexed_to == 2023
        else []
    )
    served = {
        stat_var: {
            (next(e.ref for e in s.entities if e.role == "destination"), p.date)
            for s in observations.series
            if s.stat_var.ref == stat_var
            for p in s.points
        }
        for stat_var in (flow_a, flow_b)
    }

    datasets = {stat_var: _datasets_served(world, stat_var) for stat_var in (flow_a, flow_b)}

    assert served == datasets == {flow_a: {(BLAND, "2023")}, flow_b: {(ALAND, "2018")}}
    assert spec.resolved.coverage.window == TimeWindow(start_year=2018, end_year=2023)


def _read(*dates: str) -> _SeriesRead:
    return _SeriesRead(
        points=[(date, 1.0) for date in dates],
        total_rows=len(dates),
        selections={},
        conversion_record=None,
    )


def test_a_preview_keeps_each_statvars_latest_period_across_its_reads() -> None:
    """Each read returns its series' points in its StatVar's latest year, so the preview
    reduces them once they all return. Bland keeps its December point, Aland's June falls
    below the StatVar's period and an empty read holds none, so both drop, and another
    StatVar's 2020 is its own latest."""
    flow = "TESTFIXTURE/LatestParity/Flow"
    other = "TESTFIXTURE/LatestParity/Other"

    def key(stat_var: str, place: str) -> tuple[str, tuple[tuple[str, str], ...]]:
        return stat_var, (("debtor", place),)

    reads = {
        key(flow, ALAND): _read("2023-06"),
        key(flow, BLAND): _read("2023-06", "2023-12"),
        key(flow, "country/CCC"): _read(),
        key(other, ALAND): _read("2020"),
    }

    kept, dropped = _at_latest_periods(reads)

    assert {k: read.points for k, read in kept.items()} == {
        key(flow, BLAND): [("2023-12", 1.0)],
        key(other, ALAND): [("2020", 1.0)],
    }
    assert kept[key(flow, BLAND)].total_rows == 1
    assert dropped == (key(flow, ALAND), key(flow, "country/CCC"))
