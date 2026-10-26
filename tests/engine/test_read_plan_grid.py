"""An exhaustive grid over small alphabets of facets, years and places: on every path and
under every kind of request, coverage counts exactly the points the preview delivers, a
per-pair read dispatches the one group the variable serves across its places (`_group`,
the election stated on its own from what the places hold), and an answer is
`no_observations` exactly when no place holds a point in the years read.

Each place holds one of nine facet configurations drawn from six facets (three sources, two
units, a `Projected` series, and two pairs forming one group, one pair differing only in
observation period) and six year ranges, one past the pinned 2026 clock and one with a gap
across it. One configuration's lifetime and window elections choose different facets, and in
two the gapped facet's span overlaps windows its points miss. Two places give 81 holdings,
read under six kinds of request: on
four role paths end to end (named or discovered, per pair or bulk), on resubmits of the
per-pair role answers pinning each facet axis, and on two standard paths (per pair and
scope-wide). The role and pinned-resubmit tests run a strength-4 covering array of their
cells (`GRIDS`, `_grid_cells`), and every cell under `DFO_FULL_GRIDS=1`.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import httpx
import pytest

import search.engine.region.observations.fetch as region_fetch
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from dc_sdmx import SdmxObservation
from search.engine.clients.graph import Facet
from search.engine.errors import EngineInputError
from search.engine.extract import DateRequest
from search.engine.region import promote_finalize, resolve_spec_resubmit
from search.engine.region.observations.bulk_scope import BulkScope
from search.engine.region.observations.fetch import _build_observations
from search.engine.region.state import EngineOptions
from search.engine.retrieval.read_plan import PairPlan
from search.engine.retrieval.retrieve import Materialised, materialise
from search.engine.shape import ShapeDraft
from search.models import (
    BaseIdentity,
    CoverageBreadth,
    CoverageExact,
    IdentityBindingValue,
    ResubmitIdentityPayload,
    SpecObservations,
    TimeWindow,
)
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from tests._covering_array import Grid, uncovered
from tests.engine import _grid_cells
from tests.engine import test_role_bearing_pipeline_e2e as role_bearing
from tests.engine._fake_coverage import (
    FakeCoverageService,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    _SDMX_CSV_CONTENT_TYPE,
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
)
from tests.fixtures import FakeGraph
from tests.materialise.conftest import FakeGraphReader

_KEN, _NOR = "country/KEN", "country/NOR"
_NAMES = {"Kenya": _KEN, "Norway": _NOR}

# Each facet's provenance, measurement method and unit. `fE` shares `fA`'s, so a series
# holding both serves them as one group, which materialise collapses to one row per date.
_FACETS: dict[str, tuple[str, str | None, str]] = {
    "fA": ("dc/base/SourceA", None, "USD"),
    "fB": ("dc/base/SourceB", None, "EUR"),
    "fC": ("dc/base/SourceA", "Projected", "USD"),
    "fD": ("dc/base/SourceD", None, "USD"),
    "fE": ("dc/base/SourceA", None, "USD"),
    "fF": ("dc/base/SourceA", None, "USD"),
}
# `fF` shares `fA`'s identity but carries an observation period, which is no facet axis: a
# series holding both serves one row per date (`materialise.facet_collapse`).
_PERIODS: dict[str, str] = {"fF": "P1Y"}
# "gap" is recorded to 2015 and projected from 2027, so its span overlaps every window
# from 2016 to 2026 while it holds no point in one.
_RANGES: dict[str, Sequence[int]] = {
    "old": range(1990, 2016),
    "early": range(2010, 2020),
    "late": range(2016, 2025),
    "ahead": range(2025, 2031),
    "gap": [*range(2010, 2016), *range(2027, 2031)],
    "recent": range(2021, 2023),
}
# The last two holdings make the elections disagree: `fD` holds the most points over its
# lifetime but at most one from 2015 on, so a window election and a lifetime one choose
# different facets; and `fA` with `fE` form a two-facet group.
_HOLDINGS: list[tuple[tuple[str, str], ...]] = [
    (("fA", "early"),),
    (("fA", "late"), ("fB", "early")),
    (("fA", "early"), ("fC", "ahead")),
    (("fB", "late"), ("fC", "early")),
    (("fD", "old"), ("fB", "late")),
    (("fA", "early"), ("fE", "late")),
    (("fA", "gap"),),
    (("fA", "gap"), ("fB", "recent")),
    # Two facets of one group recorded over the same years, one with an observation period:
    # materialise keeps one row per date, so they serve each year, and the latest, once.
    (("fA", "late"), ("fF", "late")),
]
_REQUESTS: dict[str, DateRequest | None] = {
    "latest": DateRequest(window=None, latest=True),
    "closed": DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False),
    "since": DateRequest(window=TimeWindow(start_year=2018, end_year=None), latest=False),
    "until": DateRequest(window=TimeWindow(start_year=None, end_year=2017), latest=False),
    # The pinned clock reads 2026, so "since 2028" asks about the future.
    "future": DateRequest(window=TimeWindow(start_year=2028, end_year=None), latest=False),
    "dateless": None,
}
# The years each request reads, as `(first, last)` with None open: an open end stops at the
# 2026 clock unless the window starts after it. None reads every year.
_READS: dict[str, tuple[int | None, int | None] | None] = {
    "latest": None,
    "closed": (2015, 2020),
    "since": (2018, 2026),
    "until": (None, 2017),
    "future": (2028, None),
    "dateless": None,
}
_CASES = [
    pytest.param(ken, nor, request_kind, id=f"KEN{ken}-NOR{nor}-{request_kind}")
    for ken, nor in itertools.product(range(len(_HOLDINGS)), repeat=2)
    for request_kind in _REQUESTS
]

# The facet axis and value a pinned resubmit posts, by name.
_PINS: dict[str, tuple[str, str]] = {
    "provenance": ("provenance", "dc/base/SourceA"),
    "method": ("measurementMethod", "Projected"),
    "unit": ("unit", "USD"),
}


# The holdings of Kenya and Norway, as indices into `_HOLDINGS`.
_KEN_HOLDINGS = _NOR_HOLDINGS = tuple(range(len(_HOLDINGS)))


def _grid(name: str, **factors: tuple[object, ...]) -> Grid[tuple[object, ...]]:
    """A grid over the holdings, the request and `factors`, whose cells are their factor
    values. A cell's id is its test id."""
    return Grid(
        name=name,
        factors={"ken": _KEN_HOLDINGS, "nor": _NOR_HOLDINGS, "request": tuple(_REQUESTS)} | factors,
        strength=4,
        cell=lambda *values: values,
        cell_id=lambda cell: "-".join([f"KEN{cell[0]}", f"NOR{cell[1]}", *map(str, cell[2:])]),
    )


ROLE_GRID = _grid("READ_PLAN_ROLE", bulk=("per_pair", "bulk"), discovered=("named", "discovered"))
PINNED_GRID = _grid("READ_PLAN_PINNED", discovered=("named", "discovered"), pin=tuple(_PINS))
GRIDS = (ROLE_GRID, PINNED_GRID)


def _params(grid: Grid[tuple[object, ...]], committed: Sequence[str]) -> list[object]:
    """The cells of `grid` a test runs (`Grid.run`), as parameters with their ids."""
    return [pytest.param(*cell, id=grid.cell_id(cell)) for cell in grid.run(committed)]


type _Holdings = dict[str, tuple[tuple[str, str], ...]]


def _holdings(ken: int, nor: int) -> _Holdings:
    return {_KEN: _HOLDINGS[ken], _NOR: _HOLDINGS[nor]}


def _points(observations: SpecObservations) -> int:
    return sum(len(s.points) for s in observations.series)


def _sdmx_rows(
    holdings: _Holdings, *, sv: str, role: str
) -> Callable[[httpx.Request], httpx.Response]:
    """Serve each dispatched facet's rows for the bound place, or its latest under `LATEST`."""

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        body = _csv_header([role])
        place = params.get(f"c[{role}]", "")
        facet_ids = params.get("c[facetId]")
        latest = params.get("c[TIME_PERIOD]") == "LATEST"
        for facet_id, years_key in holdings.get(place, ()):
            if facet_ids is not None and facet_id not in facet_ids.split(","):
                continue
            provenance, method, unit = _FACETS[facet_id]
            fixture = ObsFacetFixture(
                facet_id=facet_id,
                provenance_id=provenance,
                measurement_method=method,
                unit=unit,
                observation_period=_PERIODS.get(facet_id),
            )
            years = _RANGES[years_key]
            for year in [years[-1]] if latest else years:
                body += _csv_row(
                    stat_var=sv, role_dcids=[place], fixture=fixture, date_str=str(year), value=1.0
                )
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    return handler


def _record_dispatches(monkeypatch) -> list[tuple[frozenset[str], frozenset[str]]]:
    """Record each per-pair read's planned group (`PlanSeries.group_ids`) and the facet ids
    materialise dispatched for it."""
    recorded: list[tuple[frozenset[str], frozenset[str]]] = []
    real = region_fetch._preview_one

    def _spy(series, *args, **kwargs):
        result = real(series, *args, **kwargs)
        recorded.append((series.group_ids, region_fetch._dispatched_facet_ids(result)))
        return result

    monkeypatch.setattr(region_fetch, "_preview_one", _spy)
    return recorded


def _places_with_data(
    holdings: _Holdings, request_kind: str, facets: frozenset[str] | None = None
) -> set[str]:
    """The places holding a point the request reads, on `facets` when given: the places a
    per-series preview must serve."""
    reads = _READS[request_kind]

    def read(year: int) -> bool:
        return reads is None or (
            (reads[0] is None or year >= reads[0]) and (reads[1] is None or year <= reads[1])
        )

    return {
        place
        for place, held in holdings.items()
        for facet_id, years in held
        if (facets is None or facet_id in facets) and any(read(y) for y in _RANGES[years])
    }


def _none_last(value: str | None) -> tuple[bool, str]:
    return value is None, value or ""


def _group(
    holdings: _Holdings, request_kind: str, facets: frozenset[str] | None = None
) -> frozenset[str]:
    """The facet ids of the one group a variable serves across the places of `holdings`,
    among `facets` when given: the group of the facet that leads on the places it holds a
    point at in the years read, then the years of them its span covers, then its lifetime
    count, with ties broken by latest date and then by source, method, unit and id. A facet
    holding no point in the years read ranks behind every facet that holds one. The group
    is every facet sharing the leader's source, method and unit."""
    reads = _READS[request_kind]
    first, last = reads if reads is not None else (None, None)

    def read(year: int) -> bool:
        return (first is None or year >= first) and (last is None or year <= last)

    by_facet: dict[str, list[Sequence[int]]] = {}
    for held in holdings.values():
        for facet_id, years in held:
            if facets is None or facet_id in facets:
                by_facet.setdefault(facet_id, []).append(_RANGES[years])
    if not by_facet:
        return frozenset()

    def rank(facet_id: str) -> tuple:
        spans = by_facet[facet_id]
        earliest = min(years[0] for years in spans)
        latest = max(years[-1] for years in spans)
        places = sum(1 for years in spans if any(read(y) for y in years))
        covered = 0
        if reads is not None and read_overlaps(earliest, latest):
            start = max(y for y in (earliest, first) if y is not None)
            end = min(y for y in (latest, last) if y is not None)
            covered = max(end - start + 1, 1)
        provenance, method, unit = _FACETS[facet_id]
        return (
            places == 0,
            -places,
            -covered,
            -sum(len(years) for years in spans),
            -latest,
            _none_last(provenance),
            _none_last(method),
            _none_last(unit),
            facet_id,
        )

    def read_overlaps(earliest: int, latest: int) -> bool:
        return (first is None or latest >= first) and (last is None or earliest <= last)

    leader = min(by_facet, key=rank)
    return frozenset(f for f in by_facet if _FACETS[f] == _FACETS[leader])


def _places_served(
    holdings: _Holdings, request_kind: str, facets: frozenset[str] | None = None
) -> set[str]:
    """The places a per-series preview must serve: those holding a point the request reads
    on the variable's one group (`_group`). A place holding points only on another facet
    serves none. Under `latest` the variable serves one period, the newest year a place
    reaches on the group, so only the places reaching it serve."""
    group = _group(holdings, request_kind, facets)
    places = _places_with_data(holdings, request_kind, group)
    if request_kind != "latest":
        return places
    last = {
        place: max(
            y for facet_id, years in holdings[place] if facet_id in group for y in _RANGES[years]
        )
        for place in places
    }
    newest = max(last.values(), default=None)
    return {place for place in places if last[place] == newest}


def _served_places(observations: SpecObservations) -> set[str]:
    return {e.ref for s in observations.series for e in s.entities if e.ref in (_KEN, _NOR)}


def _record_bulk_reads(monkeypatch) -> list[object]:
    """Record each bulk role read's plan, so a check can tell a bulk answer from a
    per-series one."""
    recorded: list[object] = []
    real = region_fetch._bulk_role_series

    def _spy(plan, *args, **kwargs):
        recorded.append(plan)
        return real(plan, *args, **kwargs)

    monkeypatch.setattr(region_fetch, "_bulk_role_series", _spy)
    return recorded


_EMPTY_OMISSIONS = {"observations_nothing_to_probe", "observations_no_matching_source"}


def _assert_agreement(coverage, observations: SpecObservations | None, served_places) -> None:
    assert observations is not None
    if observations.omitted is not None:
        # Nothing to read, or a pin no read series carries: the preview serves nothing.
        assert observations.omitted.reason in _EMPTY_OMISSIONS, observations
        assert _points(observations) == 0
    if isinstance(coverage, CoverageExact):
        assert coverage.observation_count == _points(observations)
    elif isinstance(coverage, CoverageBreadth):
        assert coverage.dimensions[0].count == len(served_places(observations))
    else:
        assert not coverage.has_data
        assert _points(observations) == 0


# ---------------------------------------------------------------------------
# Role paths, end to end.
# ---------------------------------------------------------------------------


def _graph_obs(holdings: _Holdings, sv: str) -> dict[str, list[dict]]:
    """The graph's facets for each `(sv, place)`, as `FakeGraph(obs=...)` serves them."""
    return {
        f"{sv}|{place}": [
            {
                "facetId": facet_id,
                "provenanceId": _FACETS[facet_id][0],
                **({"measurementMethod": m} if (m := _FACETS[facet_id][1]) else {}),
                "unit": _FACETS[facet_id][2],
                **({"observationPeriod": p} if (p := _PERIODS.get(facet_id)) else {}),
                "earliestDate": str(_RANGES[years][0]),
                "latestDate": str(_RANGES[years][-1]),
                "obsCount": len(_RANGES[years]),
                "dates": [str(y) for y in _RANGES[years]],
                "values": [1.0] * len(_RANGES[years]),
            }
            for facet_id, years in held
        ]
        for place, held in holdings.items()
    }


@dataclass(frozen=True)
class _RoleFixture:
    coverage: FakeCoverageService
    sdmx: FakeSdmxClient
    holdings: _Holdings

    def materialise_sdmx(self) -> MaterialiseSdmxClient:
        return MaterialiseSdmxClient(
            base_url=MATERIALISE_BASE_URL,
            transport=httpx.MockTransport(
                _sdmx_rows(self.holdings, sv=role_bearing._SV_DCID, role="debtor")
            ),
        )


def _role_fixture(monkeypatch, holdings: _Holdings) -> _RoleFixture:
    sv = role_bearing._SV_DCID
    index = patch_coverage(
        monkeypatch,
        [
            (sv, facet_id, ABSENT_ENTITY, place, list(_RANGES[years]))
            for place, held in holdings.items()
            for facet_id, years in held
        ],
        role_names={sv: RoleNames(donor=None, recipient="debtor")},
        # Every DFO import states `P1Y`; a coverage facet stating no period is a cadence of
        # its own (`materialise.facets.cadences_by_member`).
        facet_metadata={
            facet_id: FacetMetadata(
                provenance_id=provenance,
                measurement_method=method,
                unit=unit,
                observation_period="P1Y",
            )
            for facet_id, (provenance, method, unit) in _FACETS.items()
        },
    )
    sdmx = FakeSdmxClient(
        observations={
            sv: {
                (("debtor", place),): [
                    Facet(
                        facet_id=facet_id,
                        provenance_id=_FACETS[facet_id][0],
                        measurement_method=_FACETS[facet_id][1],
                        unit=_FACETS[facet_id][2],
                        earliest_date=str(_RANGES[years][0]),
                        latest_date=str(_RANGES[years][-1]),
                        obs_count=len(_RANGES[years]),
                        dates=[str(y) for y in _RANGES[years]],
                        values=[1.0] * len(_RANGES[years]),
                    )
                    for facet_id, years in held
                ]
                for place, held in holdings.items()
            }
        },
        availability={},
    )
    return _RoleFixture(coverage=FakeCoverageService(index), sdmx=sdmx, holdings=holdings)


def _role_answer(fixture: _RoleFixture, *, discovered: bool, date_request: DateRequest | None):
    return role_bearing._run(
        entities=[] if discovered else list(_NAMES),
        sv_dcid=role_bearing._SV_DCID,
        sv_arcs=role_bearing._declaring_arcs(observation_properties=["debtor"]),
        resolve=_NAMES,
        sv_score=0.9,
        role_query="debt flows" if discovered else "debt flows to Kenya and Norway",
        sdmx=fixture.sdmx,
        include_observations=True,
        materialise_graph=FakeGraphReader(),
        materialise_sdmx=fixture.materialise_sdmx(),
        provenance_labels={provenance: provenance for provenance, _, _ in _FACETS.values()},
        date_request=date_request,
        coverage=fixture.coverage.snapshot(),
    )


def _served_destinations(observations: SpecObservations) -> set[str]:
    return {e.ref for s in observations.series for e in s.entities if e.role == "destination"}


@pytest.mark.parametrize(
    ("ken", "nor", "request_kind", "read", "names"),
    _params(ROLE_GRID, _grid_cells.READ_PLAN_ROLE),
)
def test_a_role_answer_counts_the_points_its_preview_serves(
    monkeypatch,
    search_settings,
    ken: int,
    nor: int,
    request_kind: str,
    read: str,
    names: str,
) -> None:
    """Runs the committed cells of `ROLE_GRID`, or every cell under `DFO_FULL_GRIDS=1`."""
    bulk, discovered = read == "bulk", names == "discovered"
    search_settings(obs_preview_max_pairs=1 if bulk else 120)
    fixture = _role_fixture(monkeypatch, _holdings(ken, nor))
    dispatches = _record_dispatches(monkeypatch)
    bulk_reads = _record_bulk_reads(monkeypatch)
    result = _role_answer(fixture, discovered=discovered, date_request=_REQUESTS[request_kind])
    # An answer is `no_observations` exactly when no place holds a point in its years.
    assert (result.status == "definite") == bool(
        _places_with_data(fixture.holdings, request_kind)
    ), result
    if result.status != "definite":
        assert result.no_data_reason == "no_observations", result
        return
    spec = result.specs[0]

    _assert_agreement(spec.resolved.coverage, spec.resolved.observations, _served_destinations)
    assert all(dispatched <= planned for planned, dispatched in dispatches)
    group = _group(fixture.holdings, request_kind)
    assert len(bulk_reads) <= 1
    if bulk_reads:
        # A bulk read asks, in each call, for the facets of the variable's one group holding
        # a point in the years read, as its mint dispatches them, and for the facets
        # continuing its lead. A bulk cell holding one pair reads it per series.
        assert bulk
        holding = {f for f in group if _places_with_data(fixture.holdings, request_kind, {f})}
        asked = holding | set(bulk_reads[0].continuing)
        assert fixture.sdmx.series_calls
        assert all(set(call["facet_ids"]) == asked for call in fixture.sdmx.series_calls)
    else:
        assert all(len(call["facet_ids"]) == 1 for call in fixture.sdmx.series_calls)
        # Every per-series read plans facets of the variable's one group alone.
        assert all(planned and planned <= group for planned, _dispatched in dispatches)
    # The preview serves each place holding a point on the group in the years read.
    assert _served_places(spec.resolved.observations) == _places_served(
        fixture.holdings, request_kind
    )


# ---------------------------------------------------------------------------
# Resubmits that pin a facet axis.
# ---------------------------------------------------------------------------


def _pinned(axis: str, value: str) -> frozenset[str]:
    position = {"provenance": 0, "measurementMethod": 1, "unit": 2}[axis]
    return frozenset(f for f, metadata in _FACETS.items() if metadata[position] == value)


@pytest.mark.parametrize(
    ("ken", "nor", "request_kind", "names", "pin"),
    _params(PINNED_GRID, _grid_cells.READ_PLAN_PINNED),
)
def test_a_pinned_resubmit_counts_the_points_its_preview_serves(
    monkeypatch, ken: int, nor: int, request_kind: str, names: str, pin: str
) -> None:
    """Runs the committed cells of `PINNED_GRID`, or every cell under `DFO_FULL_GRIDS=1`."""
    discovered = names == "discovered"
    fixture = _role_fixture(monkeypatch, _holdings(ken, nor))
    fresh = _role_answer(fixture, discovered=discovered, date_request=_REQUESTS[request_kind])
    if fresh.status != "definite":
        return
    fresh_spec = fresh.specs[0]
    axis, value = _PINS[pin]
    echo = ResubmitIdentityPayload(
        identity=BaseIdentity.model_validate(
            {
                **fresh_spec.definition.identity.model_dump(mode="python"),
                "facets": {axis: IdentityBindingValue(ref=value)},
            }
        ),
    )
    graph = FakeGraph(
        nodes={
            **{dcid: {"label": name, "type": "Country"} for name, dcid in _NAMES.items()},
            role_bearing._SV_DCID: {
                "label": "Test flow",
                "type": "StatisticalVariable",
                "arcs": role_bearing._declaring_arcs(observation_properties=["debtor"]),
            },
            **{provenance: {"label": provenance} for provenance, _, _ in _FACETS.values()},
        },
        detect={},
        resolve={},
        obs=_graph_obs(fixture.holdings, role_bearing._SV_DCID),
    )
    dispatches = _record_dispatches(monkeypatch)
    bulk_reads = _record_bulk_reads(monkeypatch)
    previews: list[object] = []
    real_build = promote_finalize._build_observations

    def _build(plan, *args, **kwargs):
        previews.append(plan)
        return real_build(plan, *args, **kwargs)

    monkeypatch.setattr(promote_finalize, "_build_observations", _build)
    try:
        region = resolve_spec_resubmit(
            inp=echo,
            graph=graph,
            sdmx=fixture.sdmx,
            coverage=fixture.coverage.snapshot(),
            # The materialise graph holds the flow's StatVar, so the resubmit's member walk
            # elects it.
            options=EngineOptions(
                include_observations=True,
                materialise_graph=FakeGraphReader(arcs=graph.member_arcs()),
                materialise_sdmx=fixture.materialise_sdmx(),
            ),
        )
    except EngineInputError as refusal:
        region = refusal
    pinned = _pinned(axis, value)
    # Method and unit pins require a facet in the years read. A provenance pin instead
    # admits a lifetime source, then reports no observations when it has no point in this
    # window.
    answers = bool(_places_with_data(fixture.holdings, request_kind, pinned))
    lifetime_present = bool(_places_with_data(fixture.holdings, "dateless", pinned))
    if axis == "provenance" and lifetime_present and not answers:
        assert not isinstance(region, EngineInputError), region
        assert region.status == "no_data"
        assert region.no_data_reason == "no_observations"
        assert region.specs == ()
        assert previews == [], "a no-data source pin fetches no preview"
        return
    if isinstance(region, EngineInputError):
        assert not answers, region
        assert region.code == "facet_not_available"
        assert previews == [], "a refused pin fetches no preview"
        return
    assert answers and region.status == "definite", region
    spec = region.specs[0]

    _assert_agreement(spec.resolved.coverage, spec.resolved.observations, _served_destinations)
    assert all(dispatched <= planned & pinned for planned, dispatched in dispatches)
    if not bulk_reads:
        # Every per-series read plans facets of the one group the variable serves among the
        # pinned facets alone, and the preview serves each place holding a point on it in the
        # years read.
        group = _group(fixture.holdings, request_kind, pinned)
        assert all(planned and planned <= group for planned, _dispatched in dispatches)
        assert _served_places(spec.resolved.observations) == _places_served(
            fixture.holdings, request_kind, pinned
        )
    served_facets = {s.facet.get("facetId") for s in spec.resolved.observations.series}
    assert served_facets <= pinned | {None}


# ---------------------------------------------------------------------------
# Standard paths, through resolve and the preview.
# ---------------------------------------------------------------------------

_SV = "TESTFIXTURE/ReadPlanGrid/Count"


def _scope_rows(holdings: _Holdings) -> dict[str, dict[str, list[SdmxObservation]]]:
    """Each place's latest row of every facet, as a scope-wide `LATEST` read returns."""
    return {
        _SV: {
            place: [
                SdmxObservation(
                    variable=_SV,
                    role_values={"observationAbout": place},
                    unit=_FACETS[facet_id][2],
                    measurement_method=_FACETS[facet_id][1],
                    observation_period=None,
                    provenance=_FACETS[facet_id][0],
                    time_period=str(_RANGES[years][-1]),
                    value="1.0",
                    scaling_factor=None,
                    facet_id=facet_id,
                )
                for facet_id, years in held
            ]
            for place, held in holdings.items()
        }
    }


@pytest.mark.parametrize("scope", [False, True], ids=["per_pair", "scope"])
@pytest.mark.parametrize(("ken", "nor", "request_kind"), _CASES)
def test_a_standard_answer_counts_the_points_its_preview_serves(
    monkeypatch, ken: int, nor: int, request_kind: str, scope: bool
) -> None:
    holdings = _holdings(ken, nor)
    obs = _graph_obs(holdings, _SV)
    graph = FakeGraph(obs=obs)
    shape = ShapeDraft(
        shape_id="test_read_plan_grid_shape",
        label="test shape",
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={_SV: {}},
        representative_score=0.9,
    )
    places = [_KEN, _NOR]
    date_request = _REQUESTS[request_kind]
    mat = materialise(
        shape,
        [],
        {} if scope else {"observationAbout": places},
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        date_request=date_request,
        place_dcids=places,
        coverage=empty_coverage_snapshot(),
    )
    assert isinstance(mat, Materialised)
    # Every plan that serves nothing yields the carrier for data outside its years, and
    # only those: every holding has data in some year.
    assert mat.has_data == bool(_places_with_data(holdings, request_kind))
    if not mat.has_data:
        # The region answers `no_observations` for the window, naming the span the series
        # runs, never a place without data.
        assert mat.available_at_place is True and mat.read_plan is None
        return
    assert mat.read_plan is not None
    materialise_sdmx = MaterialiseSdmxClient(
        base_url=MATERIALISE_BASE_URL,
        transport=httpx.MockTransport(_sdmx_rows(holdings, sv=_SV, role="observationAbout")),
    )
    dispatches = _record_dispatches(monkeypatch)
    scope_sdmx = FakeSdmxClient(subject_observations=_scope_rows(holdings))
    observations, _warn = _build_observations(
        mat.read_plan,
        mat.sv_dcids,
        {"observationAbout": places},
        graph=graph,
        materialise_graph=FakeGraphReader(),
        materialise_sdmx=materialise_sdmx,
        scope=BulkScope(places=tuple(places), sdmx=scope_sdmx) if scope else None,
        coverage=empty_coverage_snapshot(),
    )

    if scope and isinstance(mat.read_plan.years, TimeWindow):
        # A scope-wide read returns one latest point per place, so it refuses a range.
        assert observations.omitted is not None
        assert observations.omitted.reason == "observations_windowed_scope_unsupported"
        assert scope_sdmx.subject_observations_calls == []
    elif scope:
        # The scope read serves each place's latest point of the facet coverage elected.
        # A dateless answer spans every year, so that preview is truncated and its
        # baseline, `total_obs`, is what coverage counts.
        coverage = mat.coverage
        assert isinstance(coverage, CoverageExact)
        served = [p for s in observations.series for p in s.points]
        assert len(served) == len(observations.series)  # one point per place served
        if date_request is None:
            assert observations.truncated
            assert coverage.observation_count == observations.total_obs
        else:
            assert coverage.observation_count == len(served)
        assert all(
            call["facet_ids"] == [mat.read_plan.scope_facet_id]
            for call in scope_sdmx.subject_observations_calls
        )
    else:
        _assert_agreement(mat.coverage, observations, lambda _obs: set())
        assert all(dispatched <= planned for planned, dispatched in dispatches)
        group = _group(holdings, request_kind)
        assert all(planned and planned <= group for planned, _dispatched in dispatches)
        assert _served_places(observations) == _places_served(holdings, request_kind)


@pytest.mark.parametrize("scope", [False, True], ids=["per_pair", "scope"])
def test_a_standard_window_between_recorded_and_projected_years_reads_nothing(
    monkeypatch, scope: bool
) -> None:
    """Kenya records `fA` to 2019 and projects `fC` from 2028, so "since 2020" (2020 to
    the 2026 clock) serves no point: the standard path returns the carrier for data outside
    the years read, which the region answers `no_observations` for the window."""
    monkeypatch.setitem(_RANGES, "far", range(2028, 2031))
    holdings = {_KEN: (("fA", "early"), ("fC", "far"))}
    shape = ShapeDraft(
        shape_id="test_read_plan_grid_gap",
        label="test shape",
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={_SV: {}},
        representative_score=0.9,
    )
    mat = materialise(
        shape,
        [],
        {} if scope else {"observationAbout": [_KEN]},
        graph=FakeGraph(obs=_graph_obs(holdings, _SV)),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        date_request=DateRequest(window=TimeWindow(start_year=2020), latest=False),
        place_dcids=[_KEN],
        coverage=empty_coverage_snapshot(),
    )
    assert isinstance(mat, Materialised)
    assert mat.has_data is False and mat.available_at_place is True


def test_a_scope_read_elects_the_facet_with_points_in_its_years(monkeypatch) -> None:
    """Kenya and Norway hold `fA`, recorded to 2019 and projected from 2028; Uganda holds
    `fB` for 2021 and 2022. `fA` covers more places and its span overlaps 2021 to the 2026
    clock, but it holds no point there, so the election narrows to `fB` by its points, and
    the scope read serves Uganda's two."""
    _uga = "country/UGA"
    monkeypatch.setitem(_RANGES, "gap", [*range(2010, 2020), *range(2028, 2031)])
    monkeypatch.setitem(_RANGES, "recent", range(2021, 2023))
    holdings = {_KEN: (("fA", "gap"),), _NOR: (("fA", "gap"),), _uga: (("fB", "recent"),)}
    shape = ShapeDraft(
        shape_id="test_read_plan_grid_scope_gap",
        label="test shape",
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={_SV: {}},
        representative_score=0.9,
    )
    mat = materialise(
        shape,
        [],
        {},
        graph=FakeGraph(obs=_graph_obs(holdings, _SV)),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        date_request=DateRequest(window=TimeWindow(start_year=2021), latest=False),
        place_dcids=[_KEN, _NOR, _uga],
        coverage=empty_coverage_snapshot(),
    )
    assert isinstance(mat, Materialised) and mat.has_data
    plan = mat.read_plan
    assert isinstance(plan, PairPlan) and plan.scope and plan.scope_facet_id == "fB"
    assert [s.role_tuple for s in plan.served] == [(("observationAbout", _uga),)]
    assert isinstance(mat.coverage, CoverageExact)
    assert mat.coverage.window == TimeWindow(start_year=2021, end_year=2022)


@pytest.mark.parametrize("missing", ["facetId", "provenanceId"])
@pytest.mark.parametrize(
    ("window", "has_data"),
    [
        (TimeWindow(start_year=2021), False),
        (TimeWindow(start_year=2021, end_year=2023), False),
        (TimeWindow(start_year=2015, end_year=2018), True),
    ],
    ids=["since_2021", "2021_to_2023", "2015_to_2018"],
)
def test_a_series_missing_an_id_or_a_provenance_answers_only_inside_its_years(
    window: TimeWindow, has_data: bool, missing: str
) -> None:
    """Kenya's only facet arrives with no `facetId`, which the graph names by position as the
    mixer names every facet, or with no provenance, which the read pins by its facet id.
    Either way the plan serves its points from 2010 to 2019 for "2015 to 2018"; for windows
    after 2019 they lie outside the years read, and the answer is the unavailable carrier."""
    graph_obs = _graph_obs({_KEN: (("fA", "early"),)}, _SV)
    for facet in graph_obs[f"{_SV}|{_KEN}"]:
        del facet[missing]
    shape = ShapeDraft(
        shape_id="test_read_plan_grid_undispatchable",
        label="test shape",
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={_SV: {}},
        representative_score=0.9,
    )
    mat = materialise(
        shape,
        [],
        {"observationAbout": [_KEN]},
        graph=FakeGraph(obs=graph_obs),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        date_request=DateRequest(window=window, latest=False),
        place_dcids=[_KEN],
        coverage=empty_coverage_snapshot(),
    )
    assert isinstance(mat, Materialised)
    assert mat.has_data is has_data
    if has_data:
        plan = mat.read_plan
        assert isinstance(plan, PairPlan)
        assert (len(plan.served), plan.unservable) == (1, ())


@pytest.mark.parametrize("grid", GRIDS, ids=[grid.name for grid in GRIDS])
def test_the_committed_cells_cover_every_four_values_of_their_grid(
    grid: Grid[tuple[object, ...]],
) -> None:
    """Every four factor values of a grid some committed cell holds, and every committed
    cell is a cell of the grid. A factor value added to a grid fails this until
    `python -m tests._covering_array` regenerates `_grid_cells`."""
    gaps = uncovered(grid, getattr(_grid_cells, grid.name))
    assert not gaps, "\n".join(gaps)
