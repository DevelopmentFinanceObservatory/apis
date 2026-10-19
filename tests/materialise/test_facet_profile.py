"""A variable's facet profile: the default it reports is what a mint naming no place, years or
facet elects (P4), its scope's countries fallback is marked (P8), and its cache keys on the
catalogue build, binds that catalogue and the caller's graph release in the flight, and stores no
failed or late flight."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from dc_graph import GraphCapExceededError, GraphReadError
from dc_sdmx import EntityFilter, SdmxClient, SdmxHttpError
from graph_generation import FakeGeneration
from graph_release.context import bound_release, bound_release_or_none
from graph_release.service import ReleaseService
from materialise import (
    EnrichmentLookup,
    MaterialisationRequest,
    facet_profile,
    materialise_spec,
    place_scope,
)
from materialise.facet_profile import (
    FacetProfile,
    FacetProfileCache,
    ProfileGap,
    ProfileUnavailable,
    read_facet_profiles,
)
from materialise.graph import Facet
from materialise.window import RequestedWindow
from origin_catalogue import BASE, DFO, Catalogue, bound_catalogue
from series_coverage.schema import FacetMetadata, RoleNames
from series_coverage.shared import CoverageSnapshot
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.graph_release.support import build_test_release, sample_tables
from tests.origin_catalogue.fake_catalogue import provenance_row, restrictive_catalogue

from ._scope_availability import ScopeAvailability
from .conftest import FakeGraphReader

_MAX_BYTES = 2 * 1024 * 1024
_ABOUT = "observationAbout"
_POPULATION = "Count_Person"
_PROFILE_READS = ThreadPoolExecutor(max_workers=2, thread_name_prefix="profile-test-reads")
_DFO_PROVENANCE = "provenance/OECDDAC1"
_BASE_PROVENANCE = "dc/base/WorldDevelopmentIndicators"
_PLACES = ("country/P1", "country/P2", "country/P3")
_ALL = RequestedWindow(kind="all")
_SDMX_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,{role},unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _facet(facet_id: str, provenance: str, *, obs_count: int = 60, first: str = "1960") -> Facet:
    return Facet(
        facet_id=facet_id,
        provenance_id=provenance,
        measurement_method="Census",
        unit="Person",
        obs_count=obs_count,
        earliest_date=first,
        latest_date="2025",
    )


_DFO = _facet("fD", _DFO_PROVENANCE)
# More observations than DFO's facet, so it leads wherever DFO precedence does not apply.
_BASE = _facet("fB", _BASE_PROVENANCE, obs_count=66)


def _overlap(*, build_id: str = "overlap") -> Catalogue:
    """Population held by DFO (OECD DAC1) and by World Development Indicators."""
    return restrictive_catalogue(
        provenances=[provenance_row(_DFO_PROVENANCE, DFO), provenance_row(_BASE_PROVENANCE, BASE)],
        dfo_variables=[_POPULATION],
        base_variables={_POPULATION: ([_BASE_PROVENANCE], [])},
        build_id=build_id,
    )


def _base_only(*, build_id: str = "base-only") -> Catalogue:
    """The same provenances, with population listed under base Data Commons alone."""
    return restrictive_catalogue(
        provenances=[provenance_row(_DFO_PROVENANCE, DFO), provenance_row(_BASE_PROVENANCE, BASE)],
        base_variables={_POPULATION: ([_BASE_PROVENANCE], [])},
        build_id=build_id,
    )


class _ProfileGraph(FakeGraphReader):
    """`FakeGraphReader` taking the facet read's byte cap, which it records. `fail` raises
    from the next facet reads it counts down; `gate` holds each facet read until it is set."""

    def __init__(self, *, fail: int = 0, gate: threading.Event | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_bytes: list[int | None] = []
        self._fail = fail
        self._gate = gate

    def observation_facets_batch_many(
        self,
        *,
        stat_vars: Sequence[str],
        entities: Sequence[str],
        deadline: float | None = None,
        max_bytes: int | None = None,
    ) -> dict[str, dict[str, list[Facet]]]:
        self.max_bytes.append(max_bytes)
        if self._gate is not None:
            self._gate.wait(5)
        if self._fail:
            self._fail -= 1
            raise GraphReadError("graph read failed")
        return super().observation_facets_batch_many(
            stat_vars=stat_vars, entities=entities, deadline=deadline
        )


class _FailingAvailability(ScopeAvailability):
    """Availability whose read of a variable in `statuses` fails with that HTTP status, as
    `Count_Person`'s does on dev with 500, until `failures` reads have failed."""

    def __init__(self, *, statuses: Mapping[str, int], failures: int = 1_000, **kwargs) -> None:
        super().__init__(**kwargs)
        self._statuses = statuses
        self._failures = failures

    def availability_values(
        self,
        *,
        variables: Sequence[str],
        component: str,
        filters: Mapping[str, EntityFilter] | None = None,
        max_values: int | None = None,
        max_bytes: int | None = None,
        deadline: float | None = None,
        read_seconds: float | None = None,
    ) -> list[str]:
        failing = [self._statuses[v] for v in variables if v in self._statuses]
        if failing and self._failures:
            self._failures -= 1
            raise SdmxHttpError("server error", upstream_status=failing[0])
        return super().availability_values(
            variables=variables, component=component, filters=filters, max_values=max_values
        )


def _graph(facets_by_place: Mapping[str, list[Facet]], **kwargs) -> _ProfileGraph:
    return _ProfileGraph(
        facets={(_POPULATION, place): facets for place, facets in facets_by_place.items()},
        child_places={("Earth", "Country"): list(_PLACES)},
        **kwargs,
    )


def _empty_coverage(build: str = "coverage") -> CoverageSnapshot:
    return FakeCoverageService(build_coverage_index([], role_names={}), stamp=build).snapshot()


def _default_ids(profile: FacetProfile | ProfileUnavailable) -> list[str | None]:
    assert isinstance(profile, FacetProfile), profile
    return [facet.facet_id for facet in profile.default]


def _sdmx(
    rows_for: Callable[[httpx.Request], str], *, role: str, availability: ScopeAvailability | None
) -> tuple[SdmxClient, list[httpx.Request]]:
    """Serves the default place scope's availability, and each data read the rows `rows_for`
    gives it. Records the data reads."""
    recorded: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if availability is not None and (response := availability.respond(request)) is not None:
            return response
        recorded.append(request)
        return httpx.Response(
            200,
            text=_SDMX_HEADER.format(role=role) + rows_for(request),
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return SdmxClient(
        base_url="https://ours.example.org", transport=httpx.MockTransport(handler)
    ), recorded


def _mint(
    request: MaterialisationRequest, *, sdmx: SdmxClient, graph: FakeGraphReader, coverage=None
):
    return materialise_spec(
        request,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties={}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )


def _dispatched(recorded: list[httpx.Request]) -> list[str]:
    return sorted(
        {
            facet_id
            for request in recorded
            for facet_id in str(request.url.params.get("c[facetId]", "")).split(",")
            if facet_id
        }
    )


def _assert_profile_is_the_mint(profile: FacetProfile, result, recorded) -> None:
    """The profile's default is the facet group the mint fetched, under the refs its trace
    records for each axis."""
    assert sorted(f.facet_id or "" for f in profile.default) == _dispatched(recorded)
    top = profile.default[0]
    refs = {entry["axis"]: entry["ref"] for entry in result.trace.facets}
    assert refs == {
        "provenance": top.provenance_id,
        "measurementMethod": top.measurement_method,
        "unit": top.unit,
    }


# --- P4: the profile's default is a no-place mint's election ------------------------------


@pytest.mark.parametrize(
    ("facets_by_place", "elected"),
    [
        # DFO holds one place of three, so the broader base facet serves (decision 14).
        pytest.param(
            {_PLACES[0]: [_DFO, _BASE], **{p: [_BASE] for p in _PLACES[1:]}},
            "fB",
            id="dfo-narrower",
        ),
        # DFO holds every place the base facet holds, so DFO leads despite fewer observations.
        pytest.param({p: [_DFO, _BASE] for p in _PLACES}, "fD", id="dfo-as-broad"),
    ],
)
def test_a_subject_only_profile_defaults_to_what_a_no_place_mint_elects(
    bind_catalogue, facets_by_place, elected
):
    bind_catalogue(_overlap())
    graph = _graph(facets_by_place)
    availability = graph.scope_availability()

    def rows_for(request: httpx.Request) -> str:
        wanted = set(str(request.url.params.get("c[facetId]", "")).split(","))
        places = str(request.url.params[f"c[{_ABOUT}]"]).split(",")
        return "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{_POPULATION},{place},{f.unit},{f.measurement_method},"
            f"P1Y,{f.provenance_id},2023,1,,{f.facet_id}\r\n"
            for place in places
            for f in facets_by_place.get(place, [])
            if f.facet_id in wanted
        )

    sdmx, recorded = _sdmx(rows_for, role=_ABOUT, availability=availability)
    request = MaterialisationRequest(
        stat_var=f"dcid:{_POPULATION}",
        role_bindings={},
        declared_roles=(_ABOUT,),
        facet_bindings={},
        window=_ALL,
    )

    result = _mint(request, sdmx=sdmx, graph=graph)
    profile = read_facet_profiles(
        [_POPULATION],
        graph=graph,
        availability=availability,
        coverage=_empty_coverage().index,
        max_bytes=_MAX_BYTES,
        deadline=None,
    )[_POPULATION]

    assert _default_ids(profile) == [elected]
    _assert_profile_is_the_mint(profile, result, recorded)
    assert profile.breadth == {
        facet_id: sum(facet_id in {f.facet_id for f in fs} for fs in facets_by_place.values())
        for facet_id in ("fD", "fB")
    }
    assert profile.places == _PLACES
    assert not profile.countries_fallback
    assert graph.max_bytes == [_MAX_BYTES]


_FLOWS = "Amount_Flows"
_RECIPIENTS = ("country/R1", "country/R2", "country/R3")


def test_a_role_bearing_profile_defaults_to_what_a_no_place_mint_elects():
    """`fWide` covers three recipients in one year; `fLong` one recipient over eleven years.
    Breadth across every role tuple elects `fWide`; a count alone would elect `fLong`."""
    coverage = build_coverage_index(
        [
            *((_FLOWS, "fWide", "country/DONOR", r, [2015]) for r in _RECIPIENTS),
            (_FLOWS, "fLong", "country/DONOR", _RECIPIENTS[0], list(range(2010, 2021))),
        ],
        role_names={_FLOWS: RoleNames(donor="donor", recipient="recipient")},
        facet_metadata={
            "fWide": FacetMetadata(provenance_id="prov/Wide", unit="USD", observation_period="P1Y"),
            "fLong": FacetMetadata(provenance_id="prov/Long", unit="USD", observation_period="P1Y"),
        },
    )

    def rows_for(request: httpx.Request) -> str:
        wanted = set(str(request.url.params.get("c[facetId]", "")).split(","))
        return "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{_FLOWS},{r},USD,,P1Y,prov/Wide,2015,1,,fWide\r\n"
            for r in _RECIPIENTS
            if "fWide" in wanted
        )

    sdmx, recorded = _sdmx(rows_for, role="recipient", availability=None)
    graph = _ProfileGraph()
    request = MaterialisationRequest(
        stat_var=f"dcid:{_FLOWS}",
        role_bindings={},
        declared_roles=("recipient",),
        facet_bindings={},
        window=_ALL,
    )

    result = _mint(request, sdmx=sdmx, graph=graph, coverage=coverage)
    profile = read_facet_profiles(
        [_FLOWS],
        graph=graph,
        availability=ScopeAvailability(data=[]),
        coverage=coverage,
        max_bytes=_MAX_BYTES,
        deadline=None,
    )[_FLOWS]

    assert _default_ids(profile) == ["fWide"]
    _assert_profile_is_the_mint(profile, result, recorded)
    assert profile.breadth == {"fWide": 3, "fLong": 1}
    assert profile.places_by_facet == {}
    # Coverage answers a role-bearing variable, so the profile reads no graph facets.
    assert graph.max_bytes == []


# --- P8: the countries fallback is marked ------------------------------------------------


@pytest.mark.parametrize(
    ("statuses", "fallback"),
    [
        pytest.param({_POPULATION: 500}, True, id="availability-500"),
        pytest.param({}, False, id="availability-lists"),
    ],
)
def test_a_scope_that_falls_back_to_countries_is_marked(bind_catalogue, statuses, fallback):
    bind_catalogue(_overlap())
    # Earth holds data too, so a full listing reaches it and the countries fallback does not.
    graph = _graph({**{p: [_BASE] for p in _PLACES}, "Earth": [_BASE]})
    availability = _FailingAvailability(
        statuses=statuses, data=[(_POPULATION, p) for p in (*_PLACES, "Earth")]
    )

    profile = read_facet_profiles(
        [_POPULATION],
        graph=graph,
        availability=availability,
        coverage=_empty_coverage().index,
        max_bytes=_MAX_BYTES,
        deadline=None,
    )[_POPULATION]

    assert profile.countries_fallback is fallback
    assert profile.places == (_PLACES if fallback else (*_PLACES, "Earth"))
    assert _default_ids(profile) == ["fB"]


# --- the cache ------------------------------------------------------------------------------


@pytest.fixture
def executor() -> Iterator[ThreadPoolExecutor]:
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="profile-test") as pool:
        yield pool


def _cache(
    graph, executor, *, availability=None, flight_timeout_s: float = 5.0
) -> FacetProfileCache:
    return FacetProfileCache(
        graph=graph,
        availability=availability or graph.scope_availability(),
        executor=executor,
        reads=_PROFILE_READS,
        generation=FakeGeneration(token="g1"),
        flight_timeout_s=flight_timeout_s,
        maxsize=16,
        max_bytes=_MAX_BYTES,
    )


# DFO and base hold every place, so DFO leads under the overlap catalogue and the base facet,
# with more observations, leads under the other.
_EVERY_PLACE_BOTH = {p: [_DFO, _BASE] for p in _PLACES}


def test_each_catalogue_build_is_its_own_entry_and_a_repeat_is_a_hit(executor):
    graph = _graph(_EVERY_PLACE_BOTH)
    cache = _cache(graph, executor)
    coverage = _empty_coverage()

    def default(catalogue: Catalogue) -> list[str | None]:
        found = cache.profiles([_POPULATION], catalogue=catalogue, coverage=coverage, deadline=None)
        return _default_ids(found[_POPULATION])

    assert default(_overlap()) == ["fD"]
    assert default(_base_only()) == ["fB"]
    assert default(_overlap()) == ["fD"]
    assert len(graph.max_bytes) == 2


def test_another_coverage_build_is_another_entry(executor):
    graph = _graph(_EVERY_PLACE_BOTH)
    cache = _cache(graph, executor)

    for build in ("coverage-1", "coverage-2", "coverage-1"):
        cache.profiles(
            [_POPULATION], catalogue=_overlap(), coverage=_empty_coverage(build), deadline=None
        )

    assert len(graph.max_bytes) == 2


def test_a_flight_reads_under_the_catalogue_it_is_keyed_on(executor):
    """The caller's context binds the overlap catalogue, which a flight would otherwise
    inherit; the flight keyed on the other catalogue elects under that one."""
    cache = _cache(_graph(_EVERY_PLACE_BOTH), executor)

    with bound_catalogue(_overlap()):
        found = cache.profiles(
            [_POPULATION], catalogue=_base_only(), coverage=_empty_coverage(), deadline=None
        )

    assert _default_ids(found[_POPULATION]) == ["fB"]


def test_a_flight_reads_under_the_callers_graph_release(executor, tmp_path: Path):
    """A flight runs in an empty context, so it binds the release its caller has bound."""
    manifest = build_test_release((tmp_path / "releases").as_uri(), tables=sample_tables())
    release = ReleaseService(
        uri=(tmp_path / "releases").as_uri(), local_dir=tmp_path / "local"
    ).snapshot()
    seen: list[str | None] = []

    class _ReleaseRecordingGraph(_ProfileGraph):
        def observation_facets_batch_many(self, **kwargs):
            bound = bound_release_or_none()
            seen.append(bound.release_id if bound is not None else None)
            return super().observation_facets_batch_many(**kwargs)

    cache = _cache(
        _ReleaseRecordingGraph(
            facets={(_POPULATION, place): [_DFO, _BASE] for place in _PLACES},
            child_places={("Earth", "Country"): list(_PLACES)},
        ),
        executor,
    )

    with bound_release(release):
        cache.profiles(
            [_POPULATION], catalogue=_overlap(), coverage=_empty_coverage(), deadline=None
        )

    assert seen and set(seen) == {manifest.release_id}


@pytest.mark.parametrize(
    ("flight_timeout_s", "wait_s"),
    [
        pytest.param(0.2, None, id="flight-deadline"),
        pytest.param(5.0, 0.2, id="caller-deadline"),
    ],
)
def test_a_late_flight_is_a_timeout_gap_and_stores_nothing(executor, flight_timeout_s, wait_s):
    gate = threading.Event()
    graph = _graph(_EVERY_PLACE_BOTH, gate=gate)
    cache = _cache(graph, executor, flight_timeout_s=flight_timeout_s)
    catalogue, coverage = _overlap(), _empty_coverage()
    try:
        late = cache.profiles(
            [_POPULATION],
            catalogue=catalogue,
            coverage=coverage,
            deadline=None if wait_s is None else time.monotonic() + wait_s,
        )
    finally:
        gate.set()

    assert late == {_POPULATION: ProfileUnavailable(stat_var=_POPULATION, gap=ProfileGap.TIMEOUT)}
    if wait_s is None:
        # The expired flight published nothing, so the next call reads again.
        found = cache.profiles([_POPULATION], catalogue=catalogue, coverage=coverage, deadline=None)
        assert _default_ids(found[_POPULATION]) == ["fD"]
        assert len(graph.max_bytes) == 2


@pytest.mark.parametrize(
    ("graph_failures", "statuses", "gap"),
    [
        pytest.param(1, {}, ProfileGap.FAILED, id="graph-read-fails"),
        # A gateway status leaves the scope a countries fallback for that read only.
        pytest.param(0, {_POPULATION: 503}, ProfileGap.UNSETTLED_SCOPE, id="scope-unsettled"),
    ],
)
def test_a_failed_flight_is_a_gap_and_the_next_call_reads_again(
    executor, graph_failures, statuses, gap
):
    graph = _graph(_EVERY_PLACE_BOTH, fail=graph_failures)
    availability = _FailingAvailability(
        statuses=statuses, failures=1, data=[(_POPULATION, p) for p in _PLACES]
    )
    cache = _cache(graph, executor, availability=availability)
    catalogue, coverage = _overlap(), _empty_coverage()

    failed = cache.profiles([_POPULATION], catalogue=catalogue, coverage=coverage, deadline=None)
    profile = cache.profiles([_POPULATION], catalogue=catalogue, coverage=coverage, deadline=None)[
        _POPULATION
    ]

    assert failed == {_POPULATION: ProfileUnavailable(stat_var=_POPULATION, gap=gap)}
    assert isinstance(profile, FacetProfile)
    assert [facet.facet_id for facet in profile.default] == ["fD"]
    assert not profile.countries_fallback


# --- isolation: one variable's fault leaves its siblings standing -------------------------

_SIBLINGS = ("Var_A", "Var_B", "Var_C")


@pytest.mark.parametrize(
    ("fault", "gap"),
    [
        # Var_B's availability answers a gateway status, so its scope is unsettled.
        pytest.param("unsettled", ProfileGap.UNSETTLED_SCOPE, id="unsettled-scope"),
        # Var_B's own probe raises: a read error is transient, anything else a defect.
        pytest.param(GraphReadError("read failed"), ProfileGap.FAILED, id="own-probe-read-fails"),
        pytest.param(ValueError("bug"), ProfileGap.DEFECT, id="own-probe-defect"),
    ],
)
def test_one_failing_variable_fails_alone_and_its_siblings_are_cached(
    executor, monkeypatch, fault, gap
):
    # Var_B holds its own facet, so a probe can tell its read from its siblings'.
    held = {"Var_A": _BASE, "Var_B": _facet("fVarB", _BASE_PROVENANCE), "Var_C": _BASE}
    graph = _ProfileGraph(
        facets={(sv, place): [held[sv]] for sv in _SIBLINGS for place in _PLACES},
        child_places={("Earth", "Country"): list(_PLACES)},
    )
    availability = _FailingAvailability(
        statuses={"Var_B": 503} if fault == "unsettled" else {},
        data=[(sv, place) for sv in _SIBLINGS for place in _PLACES],
    )
    if isinstance(fault, Exception):
        real_probe = facet_profile.subject_probe

        def probe(by_entity, places):
            if any(f.facet_id == "fVarB" for facets in by_entity.values() for f in facets):
                raise fault
            return real_probe(by_entity, places)

        monkeypatch.setattr(facet_profile, "subject_probe", probe)
    flights: list[tuple[str, ...]] = []
    real_profiles = facet_profile.read_facet_profiles

    def recording_profiles(stat_vars, **kwargs):
        flights.append(tuple(stat_vars))
        return real_profiles(stat_vars, **kwargs)

    monkeypatch.setattr(facet_profile, "read_facet_profiles", recording_profiles)
    cache = _cache(graph, executor, availability=availability)
    catalogue, coverage = _overlap(), _empty_coverage()

    first = cache.profiles(_SIBLINGS, catalogue=catalogue, coverage=coverage, deadline=None)
    second = cache.profiles(_SIBLINGS, catalogue=catalogue, coverage=coverage, deadline=None)

    for outcome in (first, second):
        assert outcome["Var_B"] == ProfileUnavailable(stat_var="Var_B", gap=gap)
        assert _default_ids(outcome["Var_A"]) == _default_ids(outcome["Var_C"]) == ["fB"]
    # Each variable reads in a flight of its own. The second call finds the siblings stored
    # and reads the failed variable alone.
    assert sorted(flights[:3]) == [(sv,) for sv in _SIBLINGS]
    assert flights[3:] == [("Var_B",)]


class _PairCappedGraph(_ProfileGraph):
    """Records each facet read's variables and places, and raises the byte cap on a read
    naming more than `cap_pairs` variable and place pairs, as a read past `MAX_BYTES`
    would."""

    def __init__(self, *, cap_pairs: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cap_pairs = cap_pairs
        self.reads: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def observation_facets_batch_many(
        self,
        *,
        stat_vars: Sequence[str],
        entities: Sequence[str],
        deadline: float | None = None,
        max_bytes: int | None = None,
    ) -> dict[str, dict[str, list[Facet]]]:
        self.reads.append((tuple(stat_vars), tuple(entities)))
        if len(stat_vars) * len(entities) > self._cap_pairs:
            raise GraphCapExceededError(
                "cap", cap_kind="bytes", cap=max_bytes or 0, observed=(max_bytes or 0) + 1
            )
        return super().observation_facets_batch_many(
            stat_vars=stat_vars, entities=entities, deadline=deadline, max_bytes=max_bytes
        )


@pytest.mark.parametrize(
    ("pairs_per_read", "cap_pairs", "largest_read"),
    [
        # Three variables at three places, one place a read: three reads of three pairs.
        pytest.param(3, 9, 3, id="chunked-by-pairs"),
        # One read of all nine pairs passes the cap, so it is read again in halves, places
        # first and then variables, down to reads that fit.
        pytest.param(1_000, 2, 2, id="halved-past-the-cap"),
    ],
)
def test_a_wide_scope_reads_in_chunks_and_each_facet_keeps_every_place(
    monkeypatch, pairs_per_read, cap_pairs, largest_read
):
    """Breadth is each facet's places, so a scope read in pieces gives the profile one read
    gives: every variable's facet holds all three places, and no variable fails."""
    monkeypatch.setattr(facet_profile, "PAIRS_PER_READ", pairs_per_read)
    graph = _PairCappedGraph(
        cap_pairs=cap_pairs,
        facets={(sv, place): [_BASE] for sv in _SIBLINGS for place in _PLACES},
        child_places={("Earth", "Country"): list(_PLACES)},
    )

    with bound_catalogue(_overlap()):
        outcomes = read_facet_profiles(
            _SIBLINGS,
            graph=graph,
            availability=graph.scope_availability(),
            coverage=_empty_coverage().index,
            max_bytes=_MAX_BYTES,
            deadline=None,
        )

    served = [(svs, places) for svs, places in graph.reads if len(svs) * len(places) <= cap_pairs]
    assert max(len(svs) * len(places) for svs, places in served) == largest_read
    for sv in _SIBLINGS:
        outcome = outcomes[sv]
        assert isinstance(outcome, FacetProfile), outcome
        assert outcome.breadth == {"fB": len(_PLACES)}
        assert outcome.places_by_facet == {"fB": frozenset(_PLACES)}
    # Every pair was served by exactly one read.
    assert sorted((sv, p) for svs, places in served for sv in svs for p in places) == sorted(
        (sv, p) for sv in _SIBLINGS for p in _PLACES
    )


class _TogetherGraph(_ProfileGraph):
    """Holds each facet read until `reads` of them are in flight at once; reads made one after
    another break the barrier at its timeout."""

    def __init__(self, *, reads: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._together = threading.Barrier(reads, timeout=5.0)

    def observation_facets_batch_many(self, **kwargs) -> dict[str, dict[str, list[Facet]]]:
        self._together.wait()
        return super().observation_facets_batch_many(**kwargs)


def test_the_widest_default_scope_reads_its_chunks_side_by_side():
    """A variable at `place_scope.MAX_EVERY_PLACE` places reads every `PAIRS_PER_READ` chunk
    at once, so its flight's budget covers one read rather than one per chunk."""
    places = [f"country/P{n:04}" for n in range(place_scope.MAX_EVERY_PLACE)]
    graph = _TogetherGraph(
        reads=-(-len(places) // facet_profile.PAIRS_PER_READ),
        facets={(_POPULATION, place): [_BASE] for place in places},
        child_places={("Earth", "Country"): places},
    )

    with (
        bound_catalogue(_overlap()),
        ThreadPoolExecutor(max_workers=facet_profile.CHUNK_READS) as reads,
    ):
        outcome = read_facet_profiles(
            [_POPULATION],
            graph=graph,
            availability=graph.scope_availability(),
            coverage=_empty_coverage().index,
            max_bytes=_MAX_BYTES,
            deadline=None,
            reads=reads,
        )[_POPULATION]

    assert isinstance(outcome, FacetProfile), outcome
    assert outcome.breadth == {"fB": len(places)}


class _SlowVariableGraph(_ProfileGraph):
    """Holds every facet read naming `slow` for `delay` seconds."""

    def __init__(self, *, slow: str, delay: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self._slow = slow
        self._delay = delay

    def observation_facets_batch_many(self, **kwargs) -> dict[str, dict[str, list[Facet]]]:
        if self._slow in kwargs["stat_vars"]:
            time.sleep(self._delay)
        return super().observation_facets_batch_many(**kwargs)


def test_a_cold_batch_stores_each_variable_that_finishes_and_a_retry_reads_the_rest(
    executor, monkeypatch
):
    """A variable whose reads outlast the flight's budget times out alone: its siblings'
    flights finish and store, so a retry reads only the slow variable, and a batch of cold
    variables makes progress however many it names."""
    graph = _SlowVariableGraph(
        slow="Var_B",
        delay=1.0,
        facets={(sv, place): [_BASE] for sv in _SIBLINGS for place in _PLACES},
        child_places={("Earth", "Country"): list(_PLACES)},
    )
    flights: list[tuple[str, ...]] = []
    real_profiles = facet_profile.read_facet_profiles

    def recording_profiles(stat_vars, **kwargs):
        flights.append(tuple(stat_vars))
        return real_profiles(stat_vars, **kwargs)

    monkeypatch.setattr(facet_profile, "read_facet_profiles", recording_profiles)
    cache = _cache(graph, executor, flight_timeout_s=0.5)
    catalogue, coverage = _overlap(), _empty_coverage()

    first = cache.profiles(_SIBLINGS, catalogue=catalogue, coverage=coverage, deadline=None)
    second = cache.profiles(_SIBLINGS, catalogue=catalogue, coverage=coverage, deadline=None)

    for outcome in (first, second):
        assert outcome["Var_B"] == ProfileUnavailable(stat_var="Var_B", gap=ProfileGap.TIMEOUT)
        assert _default_ids(outcome["Var_A"]) == _default_ids(outcome["Var_C"]) == ["fB"]
    assert flights.count(("Var_A",)) == flights.count(("Var_C",)) == 1


def test_one_variable_at_one_place_past_the_cap_fails_the_read():
    graph = _PairCappedGraph(
        cap_pairs=0,
        facets={(_POPULATION, place): [_BASE] for place in _PLACES},
        child_places={("Earth", "Country"): list(_PLACES)},
    )

    with bound_catalogue(_overlap()), pytest.raises(GraphCapExceededError):
        read_facet_profiles(
            [_POPULATION],
            graph=graph,
            availability=graph.scope_availability(),
            coverage=_empty_coverage().index,
            max_bytes=_MAX_BYTES,
            deadline=None,
        )
