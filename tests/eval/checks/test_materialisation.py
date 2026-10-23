"""The materialisation check: breadth, observation counts, coverage and graph errors."""

from __future__ import annotations

import copy
import functools

import pytest

from dc_graph import Facet
from search.eval.checks import CHECKS, MODAL_ONLY_CHECKS, Result, ScoreContext, Verdict, WireSchema
from series_coverage.loader import CoverageArtifactError
from tests.conftest import base_response, minimal_spec
from tests.eval.checks.conftest import (
    make_case,
    no_members_door,
    panel,
    refusing_members_door,
    response_schema,
    schema_listing_members,
)
from tests.eval.conftest import RaisingGraphClient, StubGraphClient, make_worked_example_response

PASS, FAIL, NA, U = Verdict.PASS, Verdict.FAIL, Verdict.NA, Verdict.UNVERIFIABLE
DEFINITE = make_case({"definite": {}})


class Coverage:
    """A coverage service serving `snapshots()` on each read."""

    def __init__(self, snapshots):
        self.snapshots = snapshots

    def snapshot(self):
        return self.snapshots()


# The key a test Spec lists the StatVars its members door names under (`_members_of`).
_SERVED = "served_by_members_door"


def _members_of(resp: dict):
    """A members door answering each Spec of `resp` with the StatVars it lists under
    `_SERVED`."""

    def members(spec_id: str, definition: dict) -> list[str]:
        (spec,) = (
            part["spec"] for part in resp["parts"] if part.get("spec", {}).get("spec_id") == spec_id
        )
        return list(spec[_SERVED])

    return members


def check(resp: dict, *, graph=None, coverage=None, case=DEFINITE) -> Result:
    ctx = ScoreContext(
        schema=WireSchema.from_document({"response": response_schema()}),
        graph=graph if graph is not None else StubGraphClient(),
        coverage=Coverage(coverage) if coverage is not None else None,
        members=_members_of(resp),
    )
    return CHECKS["materialisation"](case, resp, ctx)


def reason_of(result: Result) -> str:
    """The reason an n/a result names after its part prefix."""
    return result.detail.split(": ")[1]


def test_materialisation_is_the_one_modal_only_check():
    assert set(MODAL_ONLY_CHECKS) == {"materialisation"}


@pytest.mark.parametrize(
    ("case", "verdict"),
    [
        (make_case({"candidates": {}}), NA),
        # A definite-only expectation asserts the check, which the candidates status skips.
        (DEFINITE, Verdict.UNREACHED),
    ],
)
def test_materialisation_skips_candidates(case, verdict):
    spec1 = minimal_spec("s1")
    spec2 = minimal_spec("s2")
    resp = base_response(
        status="candidates",
        parts=[
            {
                "status": "candidates",
                "candidates": {
                    "ordering": "broadest_first",
                    "max_candidates": 5,
                    "specs": [spec1, spec2],
                    "truncated": False,
                },
            }
        ],
    )
    assert check(resp, case=case).verdict == verdict


@pytest.mark.parametrize(
    ("expect", "verdict", "parts"),
    [
        # A mixed block asserts the check on the definite parts of a mixed answer.
        ({"mixed": {}}, Verdict.UNREACHED, None),
        ({"parts": [{"definite": {}}, {"definite": {}}]}, Verdict.UNREACHED, {0, 1}),
        ({"parts": [{"no_data": {}}, {"candidates": {}}]}, NA, None),
    ],
    ids=["mixed", "parts", "parts-unasserted"],
)
def test_materialisation_is_unreached_where_a_definite_or_mixed_block_goes_unpaired(
    expect, verdict, parts
):
    """crs-013 returns one no-data part."""
    result = check(panel("crs-013"), case=make_case(expect))
    assert result.verdict == verdict
    assert (set(result.parts) if result.parts is not None else None) == parts


def test_materialisation_breadth_pass():
    ev = check(make_worked_example_response())
    assert ev.verdict == PASS, ev.detail


def test_materialisation_breadth_zero_count():
    resp = make_worked_example_response()
    resp["parts"][0]["spec"]["resolved"]["coverage"]["dimensions"][0]["count"] = 0
    ev = check(resp)
    assert ev.verdict == FAIL
    assert "count<=0" in ev.detail


def test_materialisation_breadth_no_data():
    resp = make_worked_example_response()
    resp["parts"][0]["spec"]["resolved"]["coverage"]["has_data"] = False
    assert check(resp).verdict == FAIL


_WDI = "dc/base/WorldDevelopmentIndicators"
_UN = "dc/base/UN_WorldPopulationProspects"


def _facet(
    facet_id: str,
    years: range | list[int],
    *,
    provenance: str | None = _WDI,
    method: str | None = None,
    unit: str | None = None,
    period: str | None = None,
) -> Facet:
    dates = tuple(str(year) for year in years)
    return Facet(
        obs_count=len(dates),
        earliest_date=min(dates),
        latest_date=max(dates),
        facet_id=facet_id,
        provenance_id=provenance,
        measurement_method=method,
        unit=unit,
        observation_period=period,
        dates=dates,
    )


def _exact_response(
    count: int,
    *,
    facets: dict | None = None,
    window: dict | None = None,
    requested: dict | None = None,
    stat_vars: tuple[str, ...] = ("sv1",),
    entities: tuple[str, ...] = ("country/ETH",),
):
    """A definite answer stating `count` observations of `facets` (World Development
    Indicators on the provenance axis by default) for `stat_vars` at `entities`."""
    spec = copy.deepcopy(minimal_spec("s1"))
    spec["resolved"]["coverage"] = {
        "kind": "exact",
        "has_data": True,
        "observation_count": count,
        **({"window": window} if window else {}),
    }
    spec["resolved"]["facets"] = (
        {"provenance": {"basis": "fallback", "ref": _WDI}} if facets is None else facets
    )
    spec["resolved"]["stat_vars"] = {"member_count": len(stat_vars)}
    spec[_SERVED] = stat_vars
    spec["resolved"]["entities"] = [
        {"ref": entity, "label": entity, "role": "observationAbout"} for entity in entities
    ]
    if requested is not None:
        spec["definition"]["identity"]["requested_window"] = requested
    return base_response(status="definite", parts=[{"status": "definite", "spec": spec}])


def _score(resp, facets: dict[tuple[str, str], list[Facet]]) -> Result:
    return check(resp, graph=StubGraphClient(facets=facets))


_ETH = ("sv1", "country/ETH")
_KEN = ("sv1", "country/KEN")
_TWO_SOURCES = {
    _ETH: [
        _facet("f-wdi", range(2010, 2020)),
        _facet("f-un", range(2010, 2020), provenance=_UN),
    ]
}


def test_materialisation_exact_counts_the_stated_source_alone():
    """Two sources hold ten observations each and the answer serves one: ten, not twenty."""
    ev = _score(_exact_response(10), _TWO_SOURCES)
    assert ev.verdict == PASS, ev.detail
    assert "observed=10," in ev.detail


def test_materialisation_exact_counts_the_stat_vars_an_app_lists_on_its_wire():
    """An app with no members door is counted over the StatVars its own wire lists."""
    resp = _exact_response(10)
    resp["parts"][0]["spec"]["resolved"]["stat_vars"]["items"] = [{"ref": "sv1"}]
    ctx = ScoreContext(
        schema=schema_listing_members(),
        graph=StubGraphClient(facets=_TWO_SOURCES),
        members=no_members_door,
    )

    ev = CHECKS["materialisation"](DEFINITE, resp, ctx)

    assert ev.verdict == PASS, ev.detail
    assert "observed=10," in ev.detail


def test_materialisation_exact_fails_an_answer_counting_every_source():
    ev = _score(_exact_response(20), _TWO_SOURCES)
    assert ev.verdict == FAIL
    assert "count mismatch: observed=10, expected=20" in ev.detail


def test_materialisation_exact_counts_the_other_source_when_stated():
    """The count follows the provenance the answer states, whichever facet is larger."""
    facets = {
        _ETH: [
            _facet("f-wdi", range(2000, 2020)),
            _facet("f-un", range(2010, 2020), provenance=_UN),
        ]
    }
    stated = {"provenance": {"basis": "fallback", "ref": _UN}}
    assert _score(_exact_response(10, facets=stated), facets).verdict == PASS
    assert _score(_exact_response(20, facets=stated), facets).verdict == FAIL


def test_materialisation_exact_counts_overlapping_dates_of_one_group_once():
    """Two facets of one group at one observation period share 2015 to 2019: the series
    holds fifteen dates, not twenty."""
    facets = {
        _ETH: [
            _facet("f-annual", range(2005, 2020), period="P1Y"),
            _facet("f-other", range(2015, 2020), period="P1Y"),
        ]
    }
    ev = _score(_exact_response(15), facets)
    assert ev.verdict == PASS, ev.detail
    assert "observed=15," in ev.detail
    assert _score(_exact_response(20), facets).verdict == FAIL


def test_materialisation_exact_counts_one_observation_period_of_a_group():
    """A dispatch serves a series at one observation period. WHO antenatal care coverage
    at Nigeria holds four facets of one source and unit at four periods; the answer serves
    the P6Y facet's five points, which no union of the four (ten dates) states."""
    facets = {
        _ETH: [
            _facet("f-p4y", [1999, 2024], period="P4Y"),
            _facet("f-p3y", [2011, 2021], period="P3Y"),
            _facet("f-p6y", [1990, 2003, 2008, 2013, 2018], period="P6Y"),
            _facet("f-p5y", [2017], period="P5Y"),
        ]
    }
    ev = _score(_exact_response(5), facets)
    assert ev.verdict == PASS, ev.detail
    assert "observed=5," in ev.detail
    assert _score(_exact_response(10), facets).verdict == FAIL


def test_materialisation_exact_counts_inside_the_window():
    window = {"start_year": 2015, "end_year": 2017}
    ev = _score(_exact_response(3, window=window), _TWO_SOURCES)
    assert ev.verdict == PASS, ev.detail
    assert "observed=3," in ev.detail
    assert _score(_exact_response(10, window=window), _TWO_SOURCES).verdict == FAIL


def test_materialisation_exact_matches_method_and_unit():
    """A stated method and unit select the facets carrying both, and an axis the answer
    omits selects the facets declaring no value on it."""
    facets = {
        _ETH: [
            _facet("f-usd", range(2000, 2012), method="Estimate", unit="USD"),
            _facet("f-lcu", range(2000, 2030), method="Estimate", unit="LCU"),
            _facet("f-bare", range(2000, 2007)),
        ]
    }
    stated = {
        "provenance": {"basis": "fallback", "ref": _WDI},
        "measurementMethod": {"basis": "fallback", "ref": "Estimate"},
        "unit": {"basis": "bound", "ref": "USD"},
    }
    ev = _score(_exact_response(12, facets=stated), facets)
    assert ev.verdict == PASS, ev.detail
    assert "observed=12," in ev.detail
    ev = _score(_exact_response(7), facets)
    assert ev.verdict == PASS, ev.detail
    assert "observed=7," in ev.detail


def test_materialisation_exact_sums_series_and_counts_nothing_where_the_source_is_absent():
    """Kenya holds the stated source and Ethiopia holds another alone, so Ethiopia adds
    nothing."""
    facets = {
        _KEN: [
            _facet("f-wdi", range(2000, 2030)),
            _facet("f-un", range(1990, 2030), provenance=_UN),
        ],
        _ETH: [_facet("f-un", range(1990, 2030), provenance=_UN)],
    }
    resp = _exact_response(30, entities=("country/KEN", "country/ETH"))
    ev = _score(resp, facets)
    assert ev.verdict == PASS, ev.detail
    assert "observed=30," in ev.detail


@pytest.mark.parametrize(
    ("stat_vars", "window", "observed"),
    [
        pytest.param(("sv1",), (2022, 2023), 1, id="ethiopia-ends-before-the-statvar"),
        pytest.param(("sv1", "sv2"), (2022, 2023), 2, id="each-statvar-its-own-period"),
        pytest.param(("sv1",), (2022, 2022), 0, id="window-leaves-out-the-latest-period"),
    ],
)
def test_materialisation_exact_latest_counts_the_series_reaching_their_statvars_period(
    stat_vars: tuple[str, ...], window: tuple[int, int], observed: int
):
    """Under `latest` each StatVar serves its newest stated period, one point from each
    series reaching it: sv1 reaches 2023 at Kenya alone, and sv2 reaches 2022 at Ethiopia."""
    facets = {
        _KEN: [
            _facet("f-wdi", range(2000, 2024)),
            _facet("f-un", range(2000, 2026), provenance=_UN),
        ],
        _ETH: [_facet("f-wdi", range(2000, 2023))],
        ("sv2", "country/ETH"): [_facet("f-wdi", range(2000, 2023))],
    }
    entities = ("country/KEN", "country/ETH")
    start, end = window
    resp = _exact_response(
        observed,
        window={"start_year": start, "end_year": end},
        requested={"kind": "latest"},
        stat_vars=stat_vars,
        entities=entities,
    )
    ev = _score(resp, facets)
    assert ev.verdict == PASS, ev.detail
    assert f"observed={observed}," in ev.detail


def test_materialisation_exact_range_counts_every_date_in_the_window():
    """The same window read as a range holds Kenya's two dates and Ethiopia's one."""
    facets = {
        _KEN: [_facet("f-wdi", range(2000, 2024))],
        _ETH: [_facet("f-wdi", range(2000, 2023))],
    }
    window = {"start_year": 2022, "end_year": 2023}
    resp = _exact_response(3, window=window, entities=("country/KEN", "country/ETH"))
    ev = _score(resp, facets)
    assert ev.verdict == PASS, ev.detail
    assert "observed=3," in ev.detail


@pytest.mark.parametrize(
    ("estimates", "window", "observed"),
    [
        pytest.param(_WDI, (2015, 2027), 13, id="estimates-continue-past-the-series-end"),
        pytest.param(_WDI, (2015, 2024), 10, id="series-reaches-the-window-end"),
        pytest.param(_UN, (2015, 2027), 10, id="another-sources-estimates-continue-nothing"),
    ],
)
def test_materialisation_exact_counts_the_estimates_continuing_a_series(
    estimates: str, window: tuple[int, int], observed: int
):
    """Kenya's reported series runs to 2024 and an `Estimated` facet to 2030. The answer
    states no method, and serves the estimates of the reported series' own source after
    2024 inside its window, so 2015 to 2027 counts ten reported years and three
    estimated ones."""
    facets = {
        _KEN: [
            _facet("f-wdi", range(2000, 2025)),
            _facet("f-est", range(2020, 2031), provenance=estimates, method="Estimated"),
        ]
    }
    start, end = window
    resp = _exact_response(
        observed, window={"start_year": start, "end_year": end}, entities=("country/KEN",)
    )
    ev = _score(resp, facets)
    assert ev.verdict == PASS, ev.detail
    assert f"observed={observed}," in ev.detail


_PER_SERIES = {"provenance": {"basis": "fallback", "selection": "per_series"}}


def test_materialisation_exact_per_series_axis_admits_either_group():
    """An axis stated `per_series` names no value, so a series holding two groups serves
    either group's count and nothing between them."""
    facets = {
        _ETH: [
            _facet("f-wdi", range(2000, 2040)),
            _facet("f-un", range(2000, 2080), provenance=_UN),
        ]
    }
    for count in (40, 80):
        ev = _score(_exact_response(count, facets=_PER_SERIES), facets)
        assert ev.verdict == PASS, ev.detail
        assert f"observed={count}, expected={count} (within 5%), nearest of 2" in ev.detail
    for count, nearest in ((30, 40), (60, 40), (120, 80)):
        ev = _score(_exact_response(count, facets=_PER_SERIES), facets)
        assert ev.verdict == FAIL
        assert f"count mismatch: observed={nearest}, expected={count}" in ev.detail


def test_materialisation_exact_per_series_checks_the_totals_one_group_per_series_gives():
    """Two places each hold a 10-point and a 50-point group, so one group per place totals
    20, 60 or 100. A count of 40 or 80 lies between them and fails."""
    two_groups = [
        _facet("f-wdi", range(2000, 2010)),
        _facet("f-un", range(2000, 2050), provenance=_UN),
    ]
    facets = {_ETH: two_groups, _KEN: two_groups}
    entities = ("country/ETH", "country/KEN")
    for count in (20, 60, 100):
        ev = _score(_exact_response(count, facets=_PER_SERIES, entities=entities), facets)
        assert ev.verdict == PASS, ev.detail
        assert f"observed={count}, expected={count} (within 5%), nearest of 3" in ev.detail
    for count in (40, 80):
        ev = _score(_exact_response(count, facets=_PER_SERIES, entities=entities), facets)
        assert ev.verdict == FAIL
        assert f"count mismatch: observed={count - 20}, expected={count}" in ev.detail


def test_materialisation_exact_per_series_applies_the_tolerance_around_each_total():
    """The 5% tolerance applies around each reachable total: 62 is within 3.1 of 60, and 64
    is 4 from 60, past its 3.2."""
    two_groups = [
        _facet("f-wdi", range(2000, 2010)),
        _facet("f-un", range(2000, 2050), provenance=_UN),
    ]
    facets = {_ETH: two_groups, _KEN: two_groups}
    entities = ("country/ETH", "country/KEN")
    ev = _score(_exact_response(62, facets=_PER_SERIES, entities=entities), facets)
    assert ev.verdict == PASS, ev.detail
    assert "observed=60, expected=62 (within 5%)" in ev.detail
    ev = _score(_exact_response(64, facets=_PER_SERIES, entities=entities), facets)
    assert ev.verdict == FAIL
    assert "count mismatch: observed=60, expected=64, tolerance=3.2" in ev.detail


def test_materialisation_exact_unit_request_admits_every_native_unit():
    """A unit bound through the request form states the currency it converts to, which no
    native facet need carry."""
    stated = {
        "provenance": {"basis": "fallback", "ref": _WDI},
        "unit": {"basis": "bound", "ref": "CurrencyUSD", "request_basis": {"currency": "bound"}},
    }
    facets = {_ETH: [_facet("f-lcu", range(2000, 2025), unit="CurrencyETB")]}
    ev = _score(_exact_response(25, facets=stated), facets)
    assert ev.verdict == PASS, ev.detail
    assert "observed=25," in ev.detail


def test_materialisation_exact_incomplete_dates_count_each_facets_observations():
    facet = Facet(
        obs_count=30,
        earliest_date="1990",
        latest_date="2019",
        facet_id="f-wdi",
        provenance_id=_WDI,
        dates=("2018", "2019"),
        dates_complete=False,
    )
    ev = _score(_exact_response(30), {_ETH: [facet]})
    assert ev.verdict == PASS, ev.detail
    assert "observed=30," in ev.detail


def test_materialisation_exact_within_tolerance():
    ev = _score(_exact_response(100), {_ETH: [_facet("f-wdi", range(1900, 1998))]})
    assert ev.verdict == PASS, ev.detail
    assert "observed=98, expected=100 (within 5%)" in ev.detail


def test_materialisation_exact_outside_tolerance():
    ev = _score(_exact_response(100), {_ETH: [_facet("f-wdi", range(1900, 1950))]})
    assert ev.verdict == FAIL
    assert "count mismatch: observed=50, expected=100, tolerance=5.0" in ev.detail


def test_materialisation_exact_stated_source_absent_from_the_graph_fails():
    ev = _score(_exact_response(10), {_ETH: [_facet("f-un", range(2010, 2020), provenance=_UN)]})
    assert ev.verdict == FAIL
    assert "count mismatch: observed=0, expected=10" in ev.detail


def test_materialisation_exact_graph_returns_no_facets_is_unchecked():
    """No facet at any place leaves the count unverified and scores no `materialisation`, so a
    golden whose baseline scores it reports `not_scored`."""
    ev = _score(_exact_response(100), {})
    assert (ev.verdict, reason_of(ev)) == (NA, "no_records")


@pytest.mark.parametrize(
    ("stat_vars", "entities"),
    [
        pytest.param(("sv1", "sv1"), ("country/ETH",), id="stat-var-twice"),
        pytest.param(("sv1",), ("country/ETH", "country/ETH"), id="entity-twice"),
    ],
)
def test_materialisation_exact_counts_a_repeated_ref_once(
    stat_vars: tuple[str, ...], entities: tuple[str, ...]
):
    ev = _score(_exact_response(10, stat_vars=stat_vars, entities=entities), _TWO_SOURCES)
    assert ev.verdict == PASS, ev.detail
    assert "observed=10," in ev.detail


def _with_series(resp: dict, places: tuple[str, ...]) -> dict:
    """`resp` carrying an observation series for sv1 at each of `places`."""
    resp["parts"][0]["spec"]["resolved"]["observations"] = {
        "series": [
            {
                "stat_var": {"ref": "sv1"},
                "entities": [{"role": "observationAbout", "ref": place}],
                "points": [],
                "total_obs": 0,
            }
            for place in places
        ],
        "total_obs": 0,
        "truncated": False,
    }
    return resp


@pytest.mark.parametrize(
    ("warning", "observed"),
    [
        pytest.param(None, 10, id="complete-preview-narrows-to-its-series"),
        pytest.param("observations_probe_failed", 20, id="probe-failed"),
        pytest.param("observations_timed_out", 20, id="timed-out"),
    ],
)
def test_materialisation_partial_preview_counts_every_planned_place(
    warning: str | None, observed: int
):
    """A preview that delivered Ethiopia's series alone after Kenya's read failed leaves the
    stated count over both places, so the eval reads both."""
    facets = {
        _ETH: [_facet("f-wdi", range(2010, 2020))],
        _KEN: [_facet("f-wdi", range(2010, 2020))],
    }
    resp = _with_series(
        _exact_response(observed, entities=("country/ETH", "country/KEN")), ("country/ETH",)
    )
    if warning is not None:
        resp["parts"][0]["warnings"] = [{"code": warning, "severity": "warn", "message": "1"}]
    ev = _score(resp, facets)
    assert ev.verdict == PASS, ev.detail
    assert f"observed={observed}," in ev.detail


def test_materialisation_region_counts_at_the_places_its_series_serve():
    """A region answer names the region alone among its entities, and its member places in
    its observation series, where the graph is read."""
    facets = {
        _ETH: [_facet("f-wdi", range(2010, 2020))],
        _KEN: [_facet("f-wdi", range(2015, 2020))],
    }
    resp = _with_series(_exact_response(15, entities=("africa",)), ("country/ETH", "country/KEN"))
    ev = _score(resp, facets)
    assert ev.verdict == PASS, ev.detail
    assert "observed=15," in ev.detail
    unserved = _score(_exact_response(15, entities=("africa",)), facets)
    assert (unserved.verdict, reason_of(unserved)) == (NA, "no_records")


def test_materialisation_region_partial_preview_is_unchecked():
    """A region answer whose preview lost Kenya's series names Ethiopia alone on the wire, so
    its planned places are unknown and the region node, which holds an aggregate, is not
    read."""
    facets = {
        _ETH: [_facet("f-wdi", range(2010, 2020))],
        _KEN: [_facet("f-wdi", range(2015, 2020))],
        ("sv1", "africa"): [_facet("f-wdi", range(2010, 2020))],
    }
    resp = _with_series(_exact_response(15, entities=("africa",)), ("country/ETH",))
    resp["parts"][0]["warnings"] = [
        {"code": "observations_probe_failed", "severity": "warn", "message": "1"}
    ]
    ev = _score(resp, facets)
    assert (ev.verdict, reason_of(ev)) == (NA, "partial_preview")


def test_materialisation_place_less_answer_counts_over_its_scope_members():
    """An answer naming no place resolves no entity and lists the places its read serves in
    its defaulted scope, where the graph is read."""
    facets = {
        _ETH: [_facet("f-wdi", range(2010, 2020))],
        _KEN: [_facet("f-wdi", range(2015, 2020))],
    }
    resp = _exact_response(15, entities=())
    resp["parts"][0]["spec"]["resolved"]["scope"] = {
        "axis": "observationAbout",
        "members": [
            {"ref": place, "role": "observationAbout"} for place in ("country/ETH", "country/KEN")
        ],
    }
    ev = _score(resp, facets)
    assert ev.verdict == PASS, ev.detail
    assert "observed=15," in ev.detail


def _two_converted_parts(warnings_at: str) -> dict:
    """Two definite parts, s1 and s2, each converting to euros, with a
    `currency_unmapped_omitted` warning concerning s1 placed at `warnings_at`."""
    stated = {
        "provenance": {"basis": "fallback", "ref": _WDI},
        "unit": {"basis": "bound", "ref": "CurrencyEUR", "request_basis": {"currency": "bound"}},
    }
    first = _exact_response(25, facets=stated)
    second = copy.deepcopy(first["parts"][0])
    second["spec"]["spec_id"] = second["spec"]["definition"]["spec_id"] = "s2"
    first["parts"].append(second)
    warning = {"code": "currency_unmapped_omitted", "severity": "warn", "message": "1 entities"}
    if warnings_at == "first-part":
        first["parts"][0]["warnings"] = [warning]
    elif warnings_at == "envelope":
        first["diagnostics"]["warnings"] = [{**warning, "spec_id": "s1"}]
    else:
        first["parts"][1]["warnings"] = [{**warning, "spec_id": "s1"}]
    return first


@pytest.mark.parametrize("warnings_at", ["first-part", "envelope", "other-part-naming-s1"])
def test_materialisation_omission_warning_concerns_its_own_part_alone(warnings_at: str):
    """One part's conversion leaving rows out leaves its own count unverified, and the
    other part of the conjunction is still counted."""
    graph = StubGraphClient(facets={_ETH: [_facet("f-lcu", range(2000, 2025), unit="CurrencyETB")]})
    resp = _two_converted_parts(warnings_at)
    two_parts = make_case({"parts": [{"definite": {}}, {"definite": {}}]})
    resp["parts"][0]["spec"]["resolved"]["coverage"]["observation_count"] = 99
    assert check(resp, graph=graph, case=two_parts).verdict == PASS
    resp["parts"][1]["spec"]["resolved"]["coverage"]["observation_count"] = 99
    ev = check(resp, graph=graph, case=two_parts)
    assert ev.verdict == FAIL
    assert ev.detail.startswith("parts[1]")


def test_materialisation_answer_naming_no_place_is_unchecked():
    """With no entity, series place or scope member there is nowhere to read."""
    ev = _score(_exact_response(15, entities=()), _TWO_SOURCES)
    assert (ev.verdict, reason_of(ev)) == (NA, "no_places")


def test_materialisation_converted_answer_leaving_rows_out_is_unchecked():
    """A conversion that leaves rows out lowers the stated count by rows the eval cannot
    name."""
    stated = {
        "provenance": {"basis": "fallback", "ref": _WDI},
        "unit": {"basis": "bound", "ref": "CurrencyEUR", "request_basis": {"currency": "bound"}},
    }
    resp = _exact_response(5, facets=stated)
    resp["parts"][0]["warnings"] = [
        {"code": "currency_unmapped_omitted", "severity": "warn", "message": "1 entities"}
    ]
    ev = _score(resp, {_ETH: [_facet("f-lcu", range(2000, 2025), unit="CurrencyETB")]})
    assert (ev.verdict, reason_of(ev)) == (NA, "conversion_omitted")


_IDS = "dc/base/WorldBankInternationalDebtStatistics"
_IDS_SV = "Amount_FinancialPosition_AmountOutstanding_ByCreditor_Nominal"
_IDS_FACET = "ids-facet"
_COVERAGE_STAMP = "coverage-stamp"


_IDS_ROWS = (
    (_IDS_SV, _IDS_FACET, "country/CHN", "country/KEN", range(1983, 2025)),
    (_IDS_SV, _IDS_FACET, "country/FRA", "country/KEN", range(2009, 2025)),
)


def _ids_coverage(stamp: str = _COVERAGE_STAMP, rows=_IDS_ROWS, methods=None, period="P1Y"):
    """Coverage for ids-debt-to-china-kenya's StatVar, holding China's claims on Kenya, 1983
    to 2024 (42 observations, as the dev coverage build records them), and France's, 2009 to
    2024, or the `rows` given, each facet of `methods` under its measurement method and
    stating the observation period `period`."""
    from series_coverage.index import CoverageIndex
    from series_coverage.schema import CoverageManifest, FacetMetadata, RoleNames
    from series_coverage.shared import CoverageSnapshot
    from tests.series_coverage.support import coverage_table

    table, horizon = coverage_table(list(rows))
    index = CoverageIndex(
        table,
        {sv: RoleNames(donor="creditor", recipient="debtor") for sv, *_ in rows},
        {
            facet_id: FacetMetadata(
                provenance_id=_IDS, measurement_method=method, observation_period=period
            )
            for facet_id, method in (methods or {_IDS_FACET: None}).items()
        },
        horizon=horizon,
    )
    manifest = CoverageManifest(
        stamp=stamp,
        build_id=f"{stamp}-build",
        built_at="2026-10-03T00:00:00Z",
        horizon=horizon,
    )
    return CoverageSnapshot(index=index, manifest=manifest, stamp=stamp)


def _role_response(
    count: int,
    entities: tuple[tuple[str, str], ...],
    *,
    stamp=_COVERAGE_STAMP,
    stat_vars: tuple[str, ...] = (_IDS_SV,),
    window: dict | None = None,
    requested: dict | None = None,
):
    resp = _exact_response(
        count,
        facets={"provenance": {"basis": "fallback", "ref": _IDS}},
        stat_vars=stat_vars,
        window=window,
        requested=requested,
    )
    resolved = resp["parts"][0]["spec"]["resolved"]
    resolved["entities"] = [{"ref": ref, "role": role} for ref, role in entities]
    resolved["coverage_stamp"] = stamp
    return resp


_KEN_DEBTOR = ("country/KEN", "destination")
_CHN_CREDITOR = ("country/CHN", "origin")
# Read as observationAbout, Kenya's own total and China's would add to 58.
_IDS_GRAPH = {
    (_IDS_SV, "country/KEN"): [_facet("f-ids", range(2009, 2025), provenance=_IDS)],
    (_IDS_SV, "country/CHN"): [_facet("f-ids", range(1983, 2025), provenance=_IDS)],
}


@pytest.mark.parametrize(
    ("entities", "count", "verdict"),
    [
        pytest.param((_KEN_DEBTOR, _CHN_CREDITOR), 42, PASS, id="ids-debt-to-china-kenya"),
        pytest.param((_KEN_DEBTOR, _CHN_CREDITOR), 58, FAIL, id="both-roles-read-as-places"),
        pytest.param((_KEN_DEBTOR,), 58, PASS, id="open-origin-one-series-per-creditor"),
    ],
)
def test_materialisation_role_bearing_counts_each_bound_role_tuple(
    entities: tuple[tuple[str, str], ...], count: int, verdict: Verdict
):
    graph = StubGraphClient(facets=_IDS_GRAPH)
    ev = check(_role_response(count, entities), graph=graph, coverage=_ids_coverage)
    assert ev.verdict == verdict, ev.detail


_IDS_SV_B = "Amount_FinancialPosition_AmountOutstanding_ByCreditor_Concessional"


@pytest.mark.parametrize(
    ("stat_vars", "observed"),
    [
        pytest.param((_IDS_SV,), 1, id="france-ends-before-the-statvar"),
        pytest.param((_IDS_SV, _IDS_SV_B), 2, id="each-statvar-its-own-period"),
    ],
)
def test_materialisation_role_bearing_latest_counts_the_tuples_reaching_their_statvars_period(
    stat_vars: tuple[str, ...], observed: int
):
    """Under `latest` each StatVar serves its own latest year from the role tuples reaching
    it: China's claims on Kenya reach 2024 and France's end in 2020, while the second
    StatVar's China tuple ends in 2020 and still serves its own 2020. The answer's window,
    2020 to 2024, holds every tuple's latest year."""
    rows = (
        (_IDS_SV, _IDS_FACET, "country/CHN", "country/KEN", range(1983, 2025)),
        (_IDS_SV, _IDS_FACET, "country/FRA", "country/KEN", range(2009, 2021)),
        (_IDS_SV_B, _IDS_FACET, "country/CHN", "country/KEN", range(2000, 2021)),
    )
    resp = _role_response(
        observed,
        (_KEN_DEBTOR,),
        stat_vars=stat_vars,
        window={"start_year": 2020, "end_year": 2024},
        requested={"kind": "latest"},
    )
    ev = check(resp, coverage=functools.partial(_ids_coverage, rows=rows))
    assert ev.verdict == PASS, ev.detail
    assert f"observed={observed}," in ev.detail


@pytest.mark.parametrize(
    ("count", "year", "period", "verdict"),
    [
        pytest.param(85, 2023, "P1Y", PASS, id="the-series-holding-the-elected-year"),
        pytest.param(44, 2024, "P1Y", FAIL, id="the-newest-period-where-a-year-was-elected"),
        pytest.param(44, 2024, "P1M", PASS, id="a-statvar-electing-nothing-at-its-newest-period"),
        pytest.param(85, 2023, "P1M", FAIL, id="a-statvar-electing-nothing-at-an-earlier-year"),
    ],
)
def test_materialisation_role_bearing_latest_counts_the_series_holding_the_elected_year(
    count: int, year: int, period: str, verdict: Verdict
):
    """Of Kenya's 91 creditors, 44 report to 2024, 41 to 2023 and 6 to 2022. The index
    elects 2023 for the yearly facet, which 85 series hold, and no year for a monthly
    one, which serves the 44 series reaching its newest period."""
    ends = [2024] * 44 + [2023] * 41 + [2022] * 6
    rows = tuple(
        (_IDS_SV, _IDS_FACET, f"creditor/{i}", "country/KEN", range(2010, end + 1))
        for i, end in enumerate(ends)
    )
    coverage = functools.partial(_ids_coverage, rows=rows, period=period)
    assert bool(coverage().index.elected_years(variables=[_IDS_SV])) == (period == "P1Y")
    resp = _role_response(
        count,
        (_KEN_DEBTOR,),
        window={"start_year": year, "end_year": year},
        requested={"kind": "latest"},
    )
    ev = check(resp, coverage=coverage)
    assert ev.verdict == verdict, ev.detail
    if verdict == PASS:
        assert f"observed={count}," in ev.detail


@pytest.mark.parametrize(
    ("window", "observed"),
    [
        pytest.param((2015, 2026), 12, id="estimates-continue-past-the-tuples-end"),
        pytest.param((2015, 2022), 8, id="tuple-reaches-the-window-end"),
    ],
)
def test_materialisation_role_bearing_counts_the_estimates_continuing_a_tuple(
    window: tuple[int, int], observed: int
):
    """China's reported claims on Kenya run to 2022 and IDS's `Estimated` facet to 2030, so
    2015 to 2026 serves eight reported years and the four estimated years after them."""
    rows = (
        (_IDS_SV, _IDS_FACET, "country/CHN", "country/KEN", range(2010, 2023)),
        (_IDS_SV, "ids-estimated", "country/CHN", "country/KEN", range(2020, 2031)),
    )
    start, end = window
    resp = _role_response(
        observed, (_KEN_DEBTOR, _CHN_CREDITOR), window={"start_year": start, "end_year": end}
    )
    coverage = functools.partial(
        _ids_coverage, rows=rows, methods={_IDS_FACET: None, "ids-estimated": "Estimated"}
    )
    ev = check(resp, coverage=coverage)
    assert ev.verdict == PASS, ev.detail
    assert f"observed={observed}," in ev.detail


def test_materialisation_role_bearing_answer_from_another_coverage_stamp_is_unchecked():
    resp = _role_response(42, (_KEN_DEBTOR, _CHN_CREDITOR))
    ev = check(resp, coverage=functools.partial(_ids_coverage, "newer-stamp"))
    assert (ev.verdict, reason_of(ev)) == (NA, "coverage_stamp_mismatch")


def _failing_coverage():
    raise CoverageArtifactError("coverage view unreadable")


def _buggy_coverage():
    raise ValueError("a bug")


@pytest.mark.parametrize(
    ("graph", "coverage", "response"),
    [
        pytest.param(
            None,
            None,
            _role_response(42, (_KEN_DEBTOR, _CHN_CREDITOR)),
            id="role-bearing-without-coverage",
        ),
        pytest.param(
            None,
            _failing_coverage,
            _role_response(42, (_KEN_DEBTOR, _CHN_CREDITOR)),
            id="coverage-error",
        ),
        pytest.param(RaisingGraphClient(), None, _exact_response(100), id="graph-error"),
    ],
)
def test_materialisation_read_failures_are_infra(graph, coverage, response):
    """A graph or coverage failure scores nothing and marks the result as infrastructure."""
    ev = check(response, graph=graph, coverage=coverage)
    assert (ev.verdict, ev.infra) == (U, True)
    assert ev.detail.startswith("parts[0]")


@pytest.mark.parametrize(
    ("graph", "coverage", "response"),
    [
        pytest.param(
            None,
            _buggy_coverage,
            _role_response(42, (_KEN_DEBTOR, _CHN_CREDITOR)),
            id="coverage",
        ),
        pytest.param(
            RaisingGraphClient(ValueError("a bug")), None, _exact_response(100), id="graph"
        ),
    ],
)
def test_materialisation_lets_an_error_that_is_no_read_failure_propagate(graph, coverage, response):
    """A bug in the check or its clients stops the run; it is no infrastructure result."""
    with pytest.raises(ValueError, match="a bug"):
        check(response, graph=graph, coverage=coverage)


@pytest.mark.parametrize(
    ("door", "verdict", "infra", "named"),
    [
        (no_members_door, U, True, "HTTPStatusError: 404"),
        (refusing_members_door, FAIL, False, "member_not_resolvable"),
    ],
    ids=["failed_read", "refused_definition"],
)
def test_materialisation_exact_scores_the_members_door_failure(
    door, verdict, infra: bool, named: str
):
    """An exact count reads the served StatVars from the members door. A failed read is
    infrastructure, and a door refusing the definition fails the answer. Each names what
    the door answered."""
    ctx = ScoreContext(
        schema=WireSchema.from_document({"response": response_schema()}),
        graph=StubGraphClient(facets=_TWO_SOURCES),
        members=door,
    )

    ev = CHECKS["materialisation"](DEFINITE, _exact_response(10), ctx)

    assert (ev.verdict, ev.infra) == (verdict, infra), ev.detail
    assert named in ev.detail


def test_materialisation_graph_infra_error_names_the_http_status():
    from search.engine.errors import GraphInfraError

    class _Raising(StubGraphClient):
        def observation_facets_batch(self, stat_vars, entities, needs_dates=False):
            raise GraphInfraError(
                "Graph returned HTTP 500 for 'https://graph.example.org/v2/observation'"
            )

    ev = check(_exact_response(100), graph=_Raising())
    assert ev.infra
    assert "GraphInfraError: Graph returned HTTP 500" in ev.detail


def _bare_response(has_data: bool):
    spec = copy.deepcopy(minimal_spec("s1"))
    spec["resolved"]["coverage"] = {"kind": "bare", "has_data": has_data}
    return base_response(status="definite", parts=[{"status": "definite", "spec": spec}])


@pytest.mark.parametrize(("has_data", "verdict"), [(True, PASS), (False, FAIL)])
def test_materialisation_bare(has_data: bool, verdict: Verdict):
    assert check(_bare_response(has_data)).verdict == verdict


def test_materialisation_reads_the_coverage_snapshot_served_when_each_answer_is_scored():
    """A coverage view published between two answers is read by the eval as the engine
    reads it, and each answer is checked against the snapshot carrying its own stamp."""
    served = iter([_ids_coverage("before-publish"), _ids_coverage("after-publish")])
    for stamp in ("before-publish", "after-publish"):
        resp = _role_response(42, (_KEN_DEBTOR, _CHN_CREDITOR), stamp=stamp)
        ev = check(resp, coverage=lambda: next(served))
        assert ev.verdict == PASS, ev.detail
