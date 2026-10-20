"""A `latest` answer and a Datasets read of the same StatVar serve one period from the same
entities. A role-bearing StatVar serves the year the coverage index elects for it
(`series_coverage.index.CoverageIndex.elected_years`), and the newest period any of its
series holds when no row holds that year. A place-keyed StatVar, which the index does not
hold, serves the newest period any of its series holds (`materialise.window.latest_period`).

Each world holds its StatVars at Aland and Bland, whose series end in different years or
whose coverage index disagrees with SDMX. Search answers through the first door of the
answer-invariants world (`_answer_invariants_world`), and Datasets previews the same StatVar
through `materialise_preview`, which reads and reduces a `latest` window as a mint does, or
mints it through `materialise_group`.
"""

from __future__ import annotations

import logging
from typing import cast

import pytest

from materialise import (
    EnrichmentLookup,
    MaterialisationRequest,
    MaterialisationResult,
    RequestedWindow,
    materialise_group,
    materialise_preview,
)
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
    Answer,
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


def _latest_answer(world: World) -> Answer:
    """The first answer to a `latest` question about Aland and Bland, with its preview."""
    with world_module.door_loop():
        return world.first([ALAND, BLAND], DATES["latest"], preview=True, unit_text=None)


def _search_served(
    world: World, answer: Answer | None = None
) -> tuple[set[tuple[str, str]], TimeWindow | None, tuple[Warning, ...]]:
    """The `(place, period)` points the preview of `answer`, by default `_latest_answer`,
    serves under `latest`, the window its coverage states and the answer's warnings."""
    if answer is None:
        answer = _latest_answer(world)
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
    ("route", "max_pairs", "bulk_reads"),
    [
        pytest.param("discovered_pair", 120, 0, id="open-role"),
        pytest.param("discovered_bulk", 1, 1, id="open-role-bulk"),
        pytest.param("standard", 120, 0, id="place-keyed"),
    ],
)
def test_latest_uses_the_newest_period_sdmx_returns_when_coverage_is_ahead(
    route: str, max_pairs: int, bulk_reads: int, search_settings
) -> None:
    """When coverage ends in 2023 but both SDMX series end in 2022, Search serves the
    same actual latest rows Datasets does rather than dropping every returned point. The
    bulk role read finds no 2023 point among its `LATEST` rows and serves them, in its one
    read."""
    search_settings(obs_preview_max_pairs=max_pairs)
    world = _world(
        route,
        ((ALAND, _TO_2023, _TO_2022), (BLAND, _TO_2023, _TO_2022)),
    )

    answer = _latest_answer(world)
    served, _window, _warnings = _search_served(world, answer)

    assert served == _datasets_served(world) == {(ALAND, "2022"), (BLAND, "2022")}
    assert answer.recorder.counts().get("sdmx.role_series", 0) == bulk_reads


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
    """Both surfaces serve the StatVar's one period from the same places, and Search's
    coverage names the year the index dates it at.

    Where Aland's series ends in 2020 the plan never serves it. Where the index dates Aland
    at 2023 and SDMX ends it in 2022, Search reads it and drops it, and the partial warning
    and the log name it. Where SDMX holds 2024 for Bland past the index's 2023, a
    place-keyed read serves 2024 from Bland alone, drops Aland's 2023, and logs the point
    outside its coverage window. A role-bearing read elected 2023 from the index, so it
    reads Bland again at 2023 and serves both places there."""
    search_settings(obs_preview_max_pairs=120)
    world = _world(route, holdings)
    expected = {(BLAND, period)}
    if world.role and outside:
        expected, short, outside = {(ALAND, "2023"), (BLAND, "2023")}, 0, False

    with caplog.at_level(logging.WARNING):
        served, window, warnings = _search_served(world)

    assert served == _datasets_served(world) == expected
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


_TOTAL, _PART = "TESTFIXTURE/LatestParity/Total", "TESTFIXTURE/LatestParity/Part"
_CREDITORS = tuple(f"country/CR{i}" for i in range(5))


def _creditor_rows(stat_var: str, ends: tuple[int, ...]) -> tuple[Row, ...]:
    """One series of `stat_var` at Bland from each creditor, from 2015 to its year in `ends`."""
    return tuple(
        Row(stat_var, "f-ids", BLAND, tuple(range(2015, end + 1)), creditor=creditor)
        for creditor, end in zip(_CREDITORS, ends, strict=True)
    )


def _mint(world: World) -> MaterialisationResult:
    """A mint of every StatVar of `world` under `latest`, every role open."""
    graph = world.materialise_graph
    return materialise_group(
        [
            MaterialisationRequest(
                stat_var=f"dcid:{stat_var}",
                role_bindings={},
                declared_roles=world.declared_roles,
                facet_bindings={},
                window=RequestedWindow(kind="latest"),
            )
            for stat_var in world.stat_vars
        ],
        sdmx=world.materialise_sdmx,
        graph=graph,
        materialiser_build="latest-parity",
        capability_version="v1",
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        coverage=world.index,
    )


def _minted_points(result: MaterialisationResult) -> set[tuple[str, str, str, str]]:
    """The `(StatVar, creditor, debtor, period)` rows `result` holds."""
    points: set[tuple[str, str, str, str]] = set()
    for frame in result.frame_chunks():
        at = {name: frame.columns.index(name) for name in ("creditor", "debtor", "time_period")}
        stat_var = frame.columns.index("stat_var")
        points.update(
            (
                cast(str, row[stat_var]).removeprefix("dcid:"),
                cast(str, row[at["creditor"]]),
                cast(str, row[at["debtor"]]),
                cast(str, row[at["time_period"]]),
            )
            for row in frame.rows
        )
    return points


@pytest.mark.parametrize(
    "route",
    [pytest.param("discovered_pair", id="per-pair"), pytest.param("discovered_bulk", id="bulk")],
)
@pytest.mark.parametrize(
    ("ends", "years", "headline"),
    [
        pytest.param(
            {_TOTAL: (2024, 2024, 2023, 2023, 2022)},
            {_TOTAL: 2023},
            (2023, 4),
            id="one-variable",
        ),
        pytest.param(
            {_TOTAL: (2024, 2024, 2024, 2024, 2023), _PART: (2024, 2023, 2023, 2023, 2023)},
            {_TOTAL: 2024, _PART: 2023},
            (2024, 4),
            id="a-part-electing-an-earlier-year",
        ),
    ],
)
def test_search_predicts_the_year_and_the_series_a_mint_serves(
    route: str,
    ends: dict[str, tuple[int, ...]],
    years: dict[str, int],
    headline: tuple[int, int],
    search_settings,
) -> None:
    """Five creditors' series of one StatVar end in 2024, 2024, 2023, 2023 and 2022: two of
    the five holding 2022 reach 2024, so the StatVar elects 2023, which four hold. Search's
    headline names 2023 with four series, its preview serves those four points, and a mint
    of the StatVar holds the same four rows and records 2023, on the per-pair route and in
    bulk. The series at the newest period alone would serve 2024 from two.

    With a second StatVar in the part, the total holds 2024 at four of five and elects it,
    and the part holds 2024 at one of five and elects 2023. The headline names 2024 with
    the total's four series, and the preview and the mint serve the total's four rows of
    2024 and the part's five of 2023."""
    search_settings(obs_preview_max_pairs=1 if route.endswith("bulk") else 120)
    world = World(
        route=route,
        rows=tuple(
            row for stat_var, held in ends.items() for row in _creditor_rows(stat_var, held)
        ),
        stat_vars=dict.fromkeys(ends, ("latestParityAmount", None)),
    )
    expected = {
        (stat_var, creditor, BLAND, str(years[stat_var]))
        for stat_var, held in ends.items()
        for creditor, end in zip(_CREDITORS, held, strict=True)
        if end >= years[stat_var]
    }

    with world_module.door_loop():
        answer = world.first([BLAND], DATES["latest"], preview=True, unit_text=None)
    (spec,) = answer.specs
    observations = spec.resolved.observations
    assert observations is not None and observations.omitted is None
    served = {
        (
            s.stat_var.ref,
            next(e.ref for e in s.entities if e.role == "origin"),
            next(e.ref for e in s.entities if e.role == "destination"),
            p.date,
        )
        for s in observations.series
        for p in s.points
    }
    minted = _mint(world)

    assert served == _minted_points(minted) == expected
    assert {sv.removeprefix("dcid:"): y.served for sv, y in minted.latest_years.items()} == years
    assert not any(year.fell_back for year in minted.latest_years.values())
    predicted = spec.resolved.headline
    assert predicted is not None and (predicted.year, predicted.series) == headline
    assert headline == (
        max(years.values()),
        sum(1 for stat_var, *_rest in expected if years[stat_var] == max(years.values())),
    )


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


def test_a_converted_latest_read_ignores_a_rate_gap_after_its_elected_year(
    search_settings, monkeypatch
) -> None:
    """Coverage ends in 2023 while SDMX has Aland through 2024 and Bland through 2025, so
    the StatVar elects 2023 and both surfaces read each place again at it. The absent 2024
    euro rate rejects neither: Search and Datasets convert the rows of 2023 alone."""
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

    assert served == _datasets_served(world, unit_text="EUR") == {(ALAND, "2023"), (BLAND, "2023")}
    assert all(series.conversion is not None for series in observations.series)
    assert [point.value for series in observations.series for point in series.points] == [
        pytest.approx(0.9),
        pytest.approx(0.9),
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


def test_a_bulk_read_judges_a_statvars_elected_year_on_its_own_points(search_settings) -> None:
    """Flow A elects 2023 and holds it at Bland. The index dates flow B to 2024, but its
    series end in 2023 and 2016, so the `LATEST` read returns flow B a 2023 point and none
    in 2024. Flow B serves its newest period from that one read, as a Datasets read of it
    does, and flow A keeps its year."""
    search_settings(obs_preview_max_pairs=1)
    flow_a, flow_b = "TESTFIXTURE/LatestParity/FlowA", "TESTFIXTURE/LatestParity/FlowB"
    world = World(
        route="discovered_bulk",
        rows=(
            Row(flow_a, "f-ids", ALAND, _TO_2020),
            Row(flow_a, "f-ids", BLAND, _TO_2023),
            Row(flow_b, "f-ids", ALAND, _TO_2024, sdmx_years=_TO_2023),
            Row(flow_b, "f-ids", BLAND, _TO_2024, sdmx_years=tuple(range(2010, 2017))),
        ),
        stat_vars={flow_a: ("latestParityAmount", None), flow_b: ("latestParityAmount", None)},
    )

    answer = _latest_answer(world)
    (spec,) = answer.specs
    observations = spec.resolved.observations
    assert observations is not None and observations.omitted is None
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

    assert served == datasets == {flow_a: {(BLAND, "2023")}, flow_b: {(ALAND, "2023")}}
    assert answer.recorder.counts()["sdmx.role_series"] == 1


def test_a_bulk_read_falls_back_to_the_newest_period_any_returned_series_holds(
    search_settings,
) -> None:
    """The index dates the flow at Aland to 2023 from two creditors, which elects 2023,
    and to 2022 from a third. SDMX holds the first two to 2020 and the third to 2022. The
    bulk read dispatches the two pairs coverage dates in 2023, finds no 2023 point, and
    serves 2022 from the third creditor's `LATEST` row, the period a Datasets read of the
    flow serves, in its one read."""
    search_settings(obs_preview_max_pairs=1)
    flow, third = "TESTFIXTURE/LatestParity/Flow", "country/THD"
    world = World(
        route="discovered_bulk",
        rows=(
            Row(flow, "f-ids", ALAND, _TO_2023, sdmx_years=_TO_2020),
            Row(flow, "f-ids", ALAND, _TO_2023, creditor=_ALTCRED, sdmx_years=_TO_2020),
            Row(flow, "f-ids", ALAND, _TO_2022, creditor=third),
        ),
        stat_vars={flow: ("latestParityAmount", None)},
    )

    answer = _latest_answer(world)
    (spec,) = answer.specs
    observations = spec.resolved.observations
    assert observations is not None and observations.omitted is None
    served = {
        (next(e.ref for e in s.entities if e.role == "origin"), p.date)
        for s in observations.series
        for p in s.points
    }

    assert served == {(third, "2022")}
    assert _datasets_served(world) == {(ALAND, "2022")}
    assert answer.recorder.counts()["sdmx.role_series"] == 1


_CREDITORS = tuple(f"country/C{i:02d}" for i in range(5))


def _years(first: int, last: int, *, without: tuple[int, ...] = ()) -> tuple[int, ...]:
    return tuple(year for year in range(first, last + 1) if year not in without)


@pytest.mark.parametrize(
    ("flow_a_reads", "flow_a_served"),
    [
        pytest.param(None, {(c, "2024") for c in _CREDITORS}, id="both-years-held"),
        pytest.param(
            (_years(2012, 2025, without=(2024,)), _years(2012, 2022)),
            {(_CREDITORS[0], "2025")},
            id="flow-a-falls-back",
        ),
    ],
)
def test_bulk_reads_of_two_elected_years_serve_each_statvar_its_own(
    search_settings,
    flow_a_reads: tuple[tuple[int, ...], tuple[int, ...]] | None,
    flow_a_served: set[tuple[str, str]],
) -> None:
    """Flow A holds 2024 at five creditors and 2025 at one, and elects 2024. Flow B holds
    2022 at five and 2024 at one, and elects 2022. One creditor of each ends after its
    year, so after `LATEST` the bulk read asks flow B for 2022 and flow A for 2024, one
    read each. Each serves its own year alone, as a Datasets read of it does. Where SDMX
    holds flow A at 2025, 2023 and 2022 only, its year returns no row and it serves its
    `LATEST` 2025 point."""
    search_settings(obs_preview_max_pairs=1)
    flow_a, flow_b = "TESTFIXTURE/LatestParity/FlowA", "TESTFIXTURE/LatestParity/FlowB"
    lead_reads, rest_reads = flow_a_reads or (None, None)
    world = World(
        route="discovered_bulk",
        rows=(
            Row(flow_a, "f-ids", ALAND, _years(2012, 2025), _CREDITORS[0], lead_reads),
            *(
                Row(flow_a, "f-ids", ALAND, _years(2012, 2024), creditor, rest_reads)
                for creditor in _CREDITORS[1:]
            ),
            Row(flow_b, "f-ids", ALAND, _years(2012, 2024, without=(2023,)), _CREDITORS[0]),
            *(
                Row(flow_b, "f-ids", ALAND, _years(2012, 2022), creditor)
                for creditor in _CREDITORS[1:]
            ),
        ),
        stat_vars={flow_a: ("latestParityAmount", None), flow_b: ("latestParityAmount", None)},
    )

    answer = _latest_answer(world)
    (spec,) = answer.specs
    observations = spec.resolved.observations
    assert observations is not None and observations.omitted is None
    served = {
        stat_var: {
            (next(e.ref for e in s.entities if e.role == "origin"), p.date)
            for s in observations.series
            if s.stat_var.ref == stat_var
            for p in s.points
        }
        for stat_var in (flow_a, flow_b)
    }

    assert served == {flow_a: flow_a_served, flow_b: {(c, "2022") for c in _CREDITORS}}
    assert {
        stat_var: {period for _creditor, period in points} for stat_var, points in served.items()
    } == {
        stat_var: {period for _place, period in _datasets_served(world, stat_var)}
        for stat_var in (flow_a, flow_b)
    }
    assert answer.recorder.counts()["sdmx.role_series"] == 3


@pytest.mark.parametrize(
    ("route", "max_pairs"),
    [
        pytest.param("discovered_pair", 120, id="per-pair"),
        pytest.param("discovered_bulk", 1, id="bulk"),
    ],
)
def test_a_latest_read_whose_elected_year_holds_only_blank_values_serves_the_newest_valid(
    search_settings, route: str, max_pairs: int
) -> None:
    """Five creditors hold 2022 and two hold 2023 beside earlier years, so the flow elects
    2022. SDMX returns each 2022 row with an empty value. A blank row holds no observation,
    so no series holds the elected year and the flow serves its newest valid period, the
    two 2023 points, on both routes as a Datasets read of it does."""
    search_settings(obs_preview_max_pairs=max_pairs)
    flow = "TESTFIXTURE/LatestParity/Flow"
    world = World(
        route=route,
        rows=(
            *(
                Row(flow, "f-ids", ALAND, _years(2012, 2022), creditor, blank_years=(2022,))
                for creditor in _CREDITORS
            ),
            *(
                Row(flow, "f-ids", ALAND, (*_years(2012, 2021), 2023), creditor)
                for creditor in (_ALTCRED, "country/THD")
            ),
        ),
        stat_vars={flow: ("latestParityAmount", None)},
    )

    answer = _latest_answer(world)
    (spec,) = answer.specs
    observations = spec.resolved.observations
    assert observations is not None and observations.omitted is None
    served = {
        (next(e.ref for e in s.entities if e.role == "origin"), p.date, p.value)
        for s in observations.series
        for p in s.points
    }

    assert served == {(_ALTCRED, "2023", 1.0), ("country/THD", "2023", 1.0)}
    assert _datasets_served(world) == {(ALAND, "2023")}


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
