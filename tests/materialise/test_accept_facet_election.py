"""Acceptance cases for the Datasets facet election the Graph API's variable card reports.

DFO's facet leads a variable DFO and base Data Commons both hold only when it covers at least
as many requested places as the best base facet (decision 14, issue #1271), and admission
stops withholding a base facet at DFO's places. A single-place request breaks a tie on each
facet's breadth across the whole variable before the unit (P12).

The rows come from `accept-graph.md`. A case for behaviour dev lacks is strict-xfail with the
slice that builds it; a case dev already answers passes now."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime

import httpx
import pytest

from dc_graph import GraphReadError
from dc_sdmx import SdmxClient
from dfo_platform.timings import PhaseTimings, collecting, record_elapsed
from graph_generation import FakeGeneration
from materialise import MaterialisationRequest, facet_profile, materialise_preview
from materialise.errors import VariableBreadthUnreadError
from materialise.facet_profile import (
    FacetProfile,
    FacetProfileCache,
    ProfileBreadth,
    ProfileGap,
    ProfileUnavailable,
)
from materialise.facets import (
    Cadence,
    bound_variable_breadth,
    probe_collection_candidates,
)
from materialise.graph import Facet
from materialise.group_facets import group_breadth, group_variable_breadth, select_group_facets
from materialise.window import RequestedWindow
from origin_catalogue import (
    BASE,
    DFO,
    Catalogue,
    CatalogueManifest,
    CatalogueSnapshot,
    UnUnserved,
    current_catalogue,
    note_un_blocked_places,
    record_un_read_failure,
    recording_un_blocked_places,
)
from origin_catalogue.graph import admit_facets
from search.engine.degraded_reads import note_degraded_read, recording_degraded_reads
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.origin_catalogue.fake_catalogue import provenance_row, restrictive_catalogue
from tests.origin_catalogue.support import CRS, WDI, sample_tables

from ._group_support import served_alone
from .conftest import FakeGraphReader

_POPULATION = "Count_Person"
_PROFILE_READS = ThreadPoolExecutor(max_workers=2, thread_name_prefix="profile-test-reads")
_DFO_PROVENANCE = "provenance/OECDDAC1"
_BASE_PROVENANCE = "dc/base/WorldDevelopmentIndicators"
_PLACES = ("country/P1", "country/P2", "country/P3")

_DFO_FACET = Facet(
    facet_id="fD",
    provenance_id=_DFO_PROVENANCE,
    measurement_method="Census",
    unit="Person",
    obs_count=66,
    earliest_date="1960",
    latest_date="2025",
)
_BASE_FACET = Facet(
    facet_id="fB",
    provenance_id=_BASE_PROVENANCE,
    measurement_method="Census",
    unit="Person",
    obs_count=60,
    earliest_date="1960",
    latest_date="2025",
)

_SDMX_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_ALL = RequestedWindow(kind="all")


@pytest.fixture
def overlap(bind_catalogue: Callable[[Catalogue], None]) -> None:
    """Population is held by DFO (OECD DAC1) and by World Development Indicators."""
    bind_catalogue(
        restrictive_catalogue(
            provenances=[
                provenance_row(_DFO_PROVENANCE, DFO),
                provenance_row(_BASE_PROVENANCE, BASE),
            ],
            dfo_variables=[_POPULATION],
            base_variables={_POPULATION: ([_BASE_PROVENANCE], [])},
        )
    )


_NONE = "NotApplicable"
"""The SDMX data route's cell for a dimension a facet declares no value on."""


def _sdmx(facets_by_place: dict[str, list[Facet]], *, variable: str) -> SdmxClient:
    """Serves one 2023 row per (place, facet) the graph holds, for the facets and places a read
    names, and the default place scope's availability from the same pairs."""
    availability = FakeGraphReader(
        facets={(variable, place): facets for place, facets in facets_by_place.items()}
    ).scope_availability()

    def handler(request: httpx.Request) -> httpx.Response:
        if (response := availability.respond(request)) is not None:
            return response
        wanted = set(str(request.url.params.get("c[facetId]", "")).split(","))
        places = str(request.url.params.get("c[observationAbout]", "")).split(",")
        rows = [
            f"dataflow,DC:DF_OBS(1.0.0),I,{variable},{place},{facet.unit or _NONE},"
            f"{facet.measurement_method or _NONE},P1Y,{facet.provenance_id},2023,1,,"
            f"{facet.facet_id}\r\n"
            for place in places
            for facet in facets_by_place.get(place, [])
            if facet.facet_id in wanted
        ]
        return httpx.Response(
            200,
            text=_SDMX_HEADER + "".join(rows),
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return SdmxClient(base_url="https://ours.example.org", transport=httpx.MockTransport(handler))


def _graph(variable: str, facets_by_place: dict[str, list[Facet]]) -> FakeGraphReader:
    return FakeGraphReader(
        facets={(variable, place): facets for place, facets in facets_by_place.items()}
    )


def _preview(
    variable: str,
    facets_by_place: dict[str, list[Facet]],
    places: tuple[str, ...],
    *,
    graph: FakeGraphReader | None = None,
):
    graph = graph if graph is not None else _graph(variable, facets_by_place)
    request = MaterialisationRequest(
        stat_var=f"dcid:{variable}",
        role_bindings={"observationAbout": list(places)},
        declared_roles=("observationAbout",),
        facet_bindings={},
        window=_ALL,
    )
    return materialise_preview(
        request, sdmx=_sdmx(facets_by_place, variable=variable), graph=graph, limit=100
    )


# --- DFO precedence only by breadth (decision 14) ------------------------------------------


@pytest.mark.usefixtures("overlap")
@pytest.mark.parametrize(
    ("dfo_breadth", "base_breadth", "elected"),
    [
        # Population of every country: DAC1 holds 50 places, WDI 169 (consult-facets.md).
        pytest.param(50, 169, "fB", id="dfo-narrower-base-serves"),
        pytest.param(169, 169, "fD", id="dfo-as-broad-dfo-serves"),
        pytest.param(170, 169, "fD", id="dfo-broader-dfo-serves"),
    ],
)
def test_dfo_leads_only_when_it_covers_at_least_as_many_places(dfo_breadth, base_breadth, elected):
    served = served_alone(
        [_DFO_FACET, _BASE_FACET],
        stat_var=_POPULATION,
        window=None,
        breadth={"fD": dfo_breadth, "fB": base_breadth},
    )

    assert [facet.facet_id for facet in served] == [elected]


@pytest.mark.usefixtures("overlap")
def test_a_single_series_keeps_dfo_precedence():
    """With no breadth (one series) the ranking is today's: DFO's facet first."""
    served = served_alone([_DFO_FACET, _BASE_FACET], stat_var=_POPULATION, window=None)

    assert [facet.facet_id for facet in served] == ["fD"]


@pytest.mark.usefixtures("overlap")
def test_a_request_for_every_place_serves_the_broader_base_facet_from_one_source():
    """Acceptance case: a population request for every country returns the World Bank's
    countries from one source. DFO holds one of three places, the base facet all three."""
    outcome = _preview(
        _POPULATION,
        {_PLACES[0]: [_DFO_FACET, _BASE_FACET], **{p: [_BASE_FACET] for p in _PLACES[1:]}},
        _PLACES,
    )

    assert outcome.selections["provenance"].ref == _BASE_PROVENANCE
    assert outcome.selections["provenance"].selection_reason != "dfo_precedence"
    assert outcome.coverage.entity_refs["observationAbout"] == frozenset(_PLACES)


@pytest.mark.usefixtures("overlap")
def test_a_request_dfo_covers_as_broadly_serves_dfo():
    outcome = _preview(
        _POPULATION,
        {_PLACES[0]: [_DFO_FACET, _BASE_FACET], _PLACES[1]: [_DFO_FACET, _BASE_FACET]},
        _PLACES[:2],
    )

    assert outcome.selections["provenance"].ref == _DFO_PROVENANCE
    assert outcome.coverage.entity_refs["observationAbout"] == frozenset(_PLACES[:2])


def _snapshot() -> CatalogueSnapshot:
    manifest = CatalogueManifest(
        build_id="b",
        built_at=datetime(2026, 9, 28, 6, 0, tzinfo=UTC),
        build_seconds=0.0,
        identity_inputs={},
        registry_hash="",
        row_counts={},
        resident_bytes=0,
        overlaps=(),
        files={},
    )
    return CatalogueSnapshot(manifest=manifest, tables=sample_tables())


def _facet(facet_id: str, provenance_id: str) -> Facet:
    return replace(_BASE_FACET, facet_id=facet_id, provenance_id=provenance_id)


def test_admission_keeps_a_base_facet_where_dfo_also_holds_data():
    """The election chooses the origin, so admission fetches the elected base facet at DFO's
    places too."""
    facets = [_facet("fWDI", WDI), _facet("fCRS", CRS)]

    assert admit_facets(_snapshot(), facets) == facets


def test_admission_still_withholds_a_provenance_the_catalogue_does_not_admit():
    facets = [_facet("fWDI", WDI), _facet("fSDG", "dc/base/UN_SDG")]

    assert admit_facets(_snapshot(), facets) == facets[:1]


# --- the single-place tie (P12) ------------------------------------------------------------

_GDP = "ValueAdded_EconomicActivity_GrossDomesticProduction_Nominal"
_WEO = "provenance/IMFWorldEconomicOutlook"
_FRANCE = "country/FRA"
_OTHERS = ("country/DEU", "country/KEN", "country/USA")


def _weo(facet_id: str, unit: str, *, obs_count: int = 46, latest: str = "2025") -> Facet:
    return Facet(
        facet_id=facet_id,
        provenance_id=_WEO,
        measurement_method=None,
        unit=unit,
        obs_count=obs_count,
        earliest_date="1980",
        latest_date=latest,
    )


def _weo_places(eur: Facet, usd: Facet) -> dict[str, list[Facet]]:
    """US dollars at France and three other places, euros at France and Germany: the dollar
    facet is broader across the whole variable, as WEO's is (197 places against 25)."""
    return {
        _FRANCE: [eur, usd],
        _OTHERS[0]: [replace(eur, obs_count=46), replace(usd, obs_count=46)],
        **{place: [replace(usd, obs_count=46)] for place in _OTHERS[1:]},
    }


@pytest.fixture
def profile_executor() -> Iterator[ThreadPoolExecutor]:
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="profile-test") as pool:
        yield pool


def _bound_profile_breadth(
    graph: FakeGraphReader,
    executor: ThreadPoolExecutor,
    *,
    refuse_unread: bool = True,
    on_fallback: Callable[[], None] | None = None,
):
    """Binds the reader a door binds: each variable's breadth from its facet profile. A
    Datasets mint refuses on an unread breadth it needs; Search does not."""
    return bound_variable_breadth(
        ProfileBreadth(
            profiles=FacetProfileCache(
                graph=graph,
                availability=graph.scope_availability(),
                executor=executor,
                reads=_PROFILE_READS,
                generation=FakeGeneration(token="g1"),
                flight_timeout_s=5.0,
                maxsize=16,
                max_bytes=2 * 1024 * 1024,
            ),
            coverage=FakeCoverageService(build_coverage_index([], role_names={})).snapshot,
        ),
        refuse_unread=refuse_unread,
        on_fallback=on_fallback,
    )


@pytest.mark.parametrize(
    ("eur", "usd", "places", "unit", "profile_reads"),
    [
        # The two facets tie at France on years, count and latest date, and the dollar facet
        # serves more places across the whole variable, so dollars serve. Only the tie reads
        # the profile.
        pytest.param(
            _weo("fEUR", "CurrencyEUR"),
            _weo("fUSD", "CurrencyUSD"),
            (_FRANCE,),
            "CurrencyUSD",
            1,
            id="tie-broader-serves",
        ),
        # The narrower facet holds more observations at France, so no tie arises, it serves,
        # and the profile is never read.
        pytest.param(
            _weo("fEUR", "CurrencyEUR", obs_count=46),
            _weo("fUSD", "CurrencyUSD", obs_count=20),
            (_FRANCE,),
            "CurrencyEUR",
            0,
            id="narrower-more-observations-serves",
        ),
        # The narrower facet reaches a later year at France, so no tie arises.
        pytest.param(
            _weo("fEUR", "CurrencyEUR", latest="2025"),
            _weo("fUSD", "CurrencyUSD", latest="2024"),
            (_FRANCE,),
            "CurrencyEUR",
            0,
            id="narrower-more-recent-serves",
        ),
        # One candidate has nothing to tie with.
        pytest.param(
            _weo("fEUR", "CurrencyEUR"),
            None,
            (_FRANCE,),
            "CurrencyEUR",
            0,
            id="one-candidate-reads-nothing",
        ),
        # Several places rank on breadth in the request, as today, and read no profile.
        pytest.param(
            _weo("fEUR", "CurrencyEUR"),
            _weo("fUSD", "CurrencyUSD"),
            (_FRANCE, *_OTHERS),
            "CurrencyUSD",
            0,
            id="many-places-broader-serves",
        ),
    ],
)
def test_a_single_place_tie_breaks_on_whole_variable_breadth_before_the_unit(
    eur, usd, places, unit, profile_reads, profile_executor
):
    facets_by_place = (
        _weo_places(eur, usd) if usd is not None else {_FRANCE: [eur], _OTHERS[0]: [eur]}
    )
    graph = _graph(_GDP, facets_by_place)

    with _bound_profile_breadth(graph, profile_executor):
        outcome = _preview(_GDP, facets_by_place, places, graph=graph)

    assert outcome.selections["unit"].ref == unit
    served = [dict(zip(outcome.frame.columns, row, strict=True)) for row in outcome.frame.rows]
    assert {row["observationAbout"] for row in served} == set(places)
    assert {row["unit"] for row in served} == {unit}
    # Only the profile reads facets across many places in one batch.
    assert graph.calls["observation_facets_batch_many"] == profile_reads


class _FailingFacetRead(FakeGraphReader):
    def observation_facets_batch_many(self, **kwargs) -> dict[str, dict[str, list[Facet]]]:
        raise GraphReadError("graph read failed")


def _failing_profile(eur: Facet, usd: Facet) -> tuple[dict[str, list[Facet]], FakeGraphReader]:
    facets_by_place = _weo_places(eur, usd)
    graph = _FailingFacetRead(
        facets={(_GDP, place): facets for place, facets in facets_by_place.items()}
    )
    return facets_by_place, graph


def _defective_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes every profile's election raise a defect, an error that is no failed read."""

    def group_served(*args, **kwargs):
        raise ValueError("profile election bug")

    monkeypatch.setattr(facet_profile, "group_served", group_served)


@pytest.mark.parametrize(
    ("defect", "transient"),
    [
        # The profile's facet read fails: a retry reads again.
        pytest.param(False, True, id="read-fails-transient"),
        # The profile's own election raises: a retry would fail alike.
        pytest.param(True, False, id="profile-defect-not-transient"),
    ],
)
def test_a_mint_refuses_a_single_place_tie_its_unread_breadth_would_decide(
    profile_executor, monkeypatch, defect, transient
):
    """A mint stores its election, so a tie on every key before breadth, between facets of
    another unit, refuses when the breadth could not be read."""
    if defect:
        _defective_profile(monkeypatch)
        facets_by_place = _weo_places(_weo("fEUR", "CurrencyEUR"), _weo("fUSD", "CurrencyUSD"))
        graph = _graph(_GDP, facets_by_place)
    else:
        facets_by_place, graph = _failing_profile(
            _weo("fEUR", "CurrencyEUR"), _weo("fUSD", "CurrencyUSD")
        )

    with (
        _bound_profile_breadth(graph, profile_executor, refuse_unread=True),
        pytest.raises(VariableBreadthUnreadError) as refused,
    ):
        _preview(_GDP, facets_by_place, (_FRANCE,), graph=graph)

    assert set(refused.value.candidates) == {"fEUR", "fUSD"}
    assert refused.value.transient is transient


def test_a_mint_serves_when_its_unread_breadth_decides_nothing(profile_executor):
    """The euro facet holds more observations at France, so the failed breadth read would
    not have changed the election and the mint serves."""
    facets_by_place, graph = _failing_profile(
        _weo("fEUR", "CurrencyEUR", obs_count=46), _weo("fUSD", "CurrencyUSD", obs_count=20)
    )

    with _bound_profile_breadth(graph, profile_executor, refuse_unread=True):
        outcome = _preview(_GDP, facets_by_place, (_FRANCE,), graph=graph)

    assert outcome.selections["unit"].ref == "CurrencyEUR"


def test_a_single_place_tie_falls_to_the_unit_when_no_profile_can_be_read(profile_executor, caplog):
    """Without refusal, as Search binds it, a failed profile read leaves the tie to the unit's
    name, logged as a warning naming the variable and its gap, and the request is served."""
    facets_by_place, graph = _failing_profile(
        _weo("fEUR", "CurrencyEUR"), _weo("fUSD", "CurrencyUSD")
    )

    with _bound_profile_breadth(graph, profile_executor, refuse_unread=False):
        outcome = _preview(_GDP, facets_by_place, (_FRANCE,), graph=graph)

    assert outcome.selections["unit"].ref == "CurrencyEUR"
    # The reader's own warning, beside the cache's record of the failed flight.
    (warning,) = [
        r
        for r in caplog.records
        if r.name == "materialise.facet_profile" and r.funcName == "__call__"
    ]
    assert warning.levelname == "WARNING"
    assert warning.args == {_GDP: "failed"}


def test_search_falls_back_on_a_defective_profile_logged_as_an_error(
    profile_executor, monkeypatch, caplog
):
    """A profile whose own election raises is a defect: Search leaves the tie to the unit's
    name, notes the degraded read, and the defect is logged as an error naming the variable."""
    _defective_profile(monkeypatch)
    facets_by_place = _weo_places(_weo("fEUR", "CurrencyEUR"), _weo("fUSD", "CurrencyUSD"))
    graph = _graph(_GDP, facets_by_place)

    with (
        recording_degraded_reads() as record,
        _bound_profile_breadth(
            graph,
            profile_executor,
            refuse_unread=False,
            on_fallback=lambda: note_degraded_read("whole-variable breadth"),
        ),
    ):
        outcome = _preview(_GDP, facets_by_place, (_FRANCE,), graph=graph)

    assert outcome.selections["unit"].ref == "CurrencyEUR"
    assert set(record.reads) == {"whole-variable breadth"}
    # Read once for the request, though each axis and the cadence rank on the tie.
    (error,) = [r for r in caplog.records if r.levelname == "ERROR"]
    assert error.name == "materialise.facet_profile"
    assert _GDP in error.getMessage()


def test_a_group_at_one_place_breaks_its_tie_on_its_members_whole_breadth():
    """Two members probed together at France read their breadth in one call and the group's
    tie goes to the facet broader across both variables (`group_variable_breadth`)."""
    other = f"{_GDP}_PerCapita"
    eur, usd = _weo("fEUR", "CurrencyEUR"), _weo("fUSD", "CurrencyUSD")
    graph = FakeGraphReader(
        facets={
            (variable, place): facets
            for variable in (_GDP, other)
            for place, facets in _weo_places(eur, usd).items()
        }
    )
    requests = [
        MaterialisationRequest(
            stat_var=f"dcid:{variable}",
            role_bindings={"observationAbout": [_FRANCE]},
            declared_roles=("observationAbout",),
            facet_bindings={},
            window=_ALL,
        )
        for variable in (_GDP, other)
    ]
    reads: list[tuple[str, ...]] = []

    def reader(stat_vars, *, deadline):
        reads.append(tuple(stat_vars))
        return {sv: {"fUSD": 4, "fEUR": 2} for sv in stat_vars}

    with bound_variable_breadth(reader, refuse_unread=True):
        probes = probe_collection_candidates(
            requests, graph=graph, availability=graph.scope_availability(), settle=True
        )
    election = select_group_facets(
        requests,
        [probe.candidates for probe in probes],
        graph=graph,
        breadth=group_breadth(probes, window=_ALL),
        variable_breadth=group_variable_breadth(probes),
    ).election

    assert reads == [(_GDP, other)]
    assert election.selections["unit"].ref == "CurrencyUSD"


# --- a profile flight reads for every caller, so it writes into none ---------------------


class _UnGraph(FakeGraphReader):
    """Reads `_GDP` as `OriginGraphClient` reads a UN variable: it notes the places in
    `refused` for the caller's UN record, and with `un_fails` a read naming another variable
    beside it records the failure and reads `_GDP` as empty, or raises with no record bound.
    A read naming a refused place also writes into the other request-scoped records a read
    can reach: the call's phase timings and Search's degraded reads."""

    def __init__(self, *, refused: frozenset[str] = frozenset(), **kwargs) -> None:
        super().__init__(**kwargs)
        self.refused = refused
        self.un_fails = False

    def observation_facets_batch_many(
        self,
        *,
        stat_vars: Sequence[str],
        entities: Sequence[str],
        needs_dates: bool = False,
        deadline: float | None = None,
        max_bytes: int | None = None,
    ) -> dict[str, dict[str, list[Facet]]]:
        found = super().observation_facets_batch_many(
            stat_vars=stat_vars, entities=entities, needs_dates=needs_dates
        )
        if _GDP not in stat_vars:
            return found
        if self.un_fails:
            failure = GraphReadError("UN observation read failed")
            if not record_un_read_failure(
                failure, variables=[_GDP], places=entities, alongside_ours=len(stat_vars) > 1
            ):
                raise failure
            found[_GDP] = {entity: [] for entity in entities}
        if refused := [entity for entity in entities if entity in self.refused]:
            note_un_blocked_places(refused)
            record_elapsed("un refused", 0.001)
            note_degraded_read("UN observations")
        return found


def test_a_cold_single_place_mint_records_only_its_own_reads(profile_executor):
    """The flight that reads the whole variable serves every caller, so it writes into none
    of the leading caller's records. Its default-scope read names the USA, which the UN front
    door refuses, and the mint at France must not report that place as unserved, time the
    flight's reads as its own or mark itself degraded."""
    facets_by_place = _weo_places(_weo("fEUR", "CurrencyEUR"), _weo("fUSD", "CurrencyUSD"))
    graph = _UnGraph(
        refused=frozenset({"country/USA"}),
        facets={(_GDP, place): facets for place, facets in facets_by_place.items()},
    )

    timings = PhaseTimings()
    with (
        recording_un_blocked_places() as record,
        recording_degraded_reads() as degraded,
        collecting(timings),
        _bound_profile_breadth(graph, profile_executor),
    ):
        outcome = _preview(_GDP, facets_by_place, (_FRANCE,), graph=graph)

    assert outcome.selections["unit"].ref == "CurrencyUSD"
    assert record.snapshot() == UnUnserved()
    assert degraded.reads == []
    assert "un refused" not in timings.snapshot()


def test_a_failed_un_read_inside_a_flight_stores_no_profile(profile_executor):
    """A UN read that fails beside another variable's would read as empty under the caller's
    record. The flight binds none, so it fails, stores nothing and the next call reads
    again."""
    other = f"{_GDP}_PerCapita"
    facets_by_place = _weo_places(_weo("fEUR", "CurrencyEUR"), _weo("fUSD", "CurrencyUSD"))
    graph = _UnGraph(
        facets={
            (variable, place): facets
            for variable in (_GDP, other)
            for place, facets in facets_by_place.items()
        }
    )
    graph.un_fails = True
    cache = FacetProfileCache(
        graph=graph,
        availability=graph.scope_availability(),
        executor=profile_executor,
        reads=_PROFILE_READS,
        generation=FakeGeneration(token="g1"),
        flight_timeout_s=5.0,
        maxsize=16,
        max_bytes=2 * 1024 * 1024,
    )
    coverage = FakeCoverageService(build_coverage_index([], role_names={})).snapshot()

    def profile() -> FacetProfile | ProfileUnavailable:
        return cache.profiles(
            [_GDP, other], catalogue=current_catalogue(), coverage=coverage, deadline=None
        )[_GDP]

    with recording_un_blocked_places() as record:
        failed = profile()
    graph.un_fails = False
    read = profile()

    assert failed == ProfileUnavailable(stat_var=_GDP, gap=ProfileGap.FAILED)
    assert record.snapshot() == UnUnserved()
    assert isinstance(read, FacetProfile)
    assert read.breadth == {"fUSD": 4, "fEUR": 2}


# --- an unread breadth that would decide the cadence --------------------------------------


@pytest.mark.parametrize(
    ("first", "last", "refused"),
    [
        # Monthly dates and no stated period: each facet is its own cadence, and the breadth
        # would decide which one dispatches.
        pytest.param("2019-01", "2025-06", True, id="own-cadences-refuse"),
        # Bare years: both read annual, share one cadence and dispatch together.
        pytest.param("1980", "2025", False, id="one-annual-cadence-serves"),
    ],
)
def test_a_mint_refuses_a_tie_its_unread_breadth_would_decide_through_cadence(
    profile_executor, first, last, refused
):
    """Two facets of one import tie at France on every key before breadth and share
    provenance, method and unit, so only the cadence tells them apart."""
    f1 = replace(_weo("f1", "CurrencyUSD"), earliest_date=first, latest_date=last)
    f2 = replace(f1, facet_id="f2")
    facets_by_place, graph = _failing_profile(f1, f2)

    with _bound_profile_breadth(graph, profile_executor, refuse_unread=True):
        if refused:
            with pytest.raises(VariableBreadthUnreadError) as refusal:
                _preview(_GDP, facets_by_place, (_FRANCE,), graph=graph)
            assert set(refusal.value.candidates) == {"f1", "f2"}
        else:
            outcome = _preview(_GDP, facets_by_place, (_FRANCE,), graph=graph)
            assert outcome.selections["unit"].ref == "CurrencyUSD"


@pytest.mark.parametrize(
    ("first_member_monthly", "refused"),
    [
        # Only the second member dates fMonthly by month, so the first member reads both
        # facets as annual: annual is held by two members, fMonthly's own cadence by one, and
        # the member count decides before any facet order. The group serves annual.
        pytest.param(False, False, id="member-count-decides-serves"),
        # Both members date fMonthly by month, so both cadences are held by two members and
        # the first-ranked facet decides, which only the unread breadth could order.
        pytest.param(True, True, id="rank-decides-refuses"),
    ],
)
def test_a_group_refuses_only_a_cadence_its_unread_breadth_would_decide(
    first_member_monthly, refused
):
    """Two facets of one import, held by both members at France, tie on every key before
    breadth and state no period. A member reads a facet as annual unless its dates show a
    month (`cadences_by_member`). The request asks for the facets' whole span as a range, the
    one window under which the probe reads their dates."""
    window = RequestedWindow(kind="range", start_year=1980, end_year=2025)
    other = f"{_GDP}_PerCapita"
    annual = replace(_weo("fAnnual", "CurrencyUSD"), dates=("1980", "2025"), dates_complete=True)
    monthly_copy = replace(annual, facet_id="fMonthly", dates=("1980", "2000-06", "2025"))
    annual_copy = replace(annual, facet_id="fMonthly")
    graph = FakeGraphReader(
        facets={
            (_GDP, _FRANCE): [annual, monthly_copy if first_member_monthly else annual_copy],
            (other, _FRANCE): [annual, monthly_copy],
        }
    )
    requests = [
        MaterialisationRequest(
            stat_var=f"dcid:{variable}",
            role_bindings={"observationAbout": [_FRANCE]},
            declared_roles=("observationAbout",),
            facet_bindings={},
            window=window,
        )
        for variable in (_GDP, other)
    ]

    def elect():
        with bound_variable_breadth(lambda stat_vars, *, deadline: {}, refuse_unread=True):
            probes = probe_collection_candidates(
                requests, graph=graph, availability=graph.scope_availability(), settle=True
            )
            return select_group_facets(
                requests,
                [probe.candidates for probe in probes],
                graph=graph,
                breadth=group_breadth(probes, window=window),
                variable_breadth=group_variable_breadth(probes),
            ).election

    if refused:
        with pytest.raises(VariableBreadthUnreadError) as refusal:
            elect()
        assert set(refusal.value.candidates) == {"fAnnual", "fMonthly"}
    else:
        assert elect().cadence == Cadence(period="P1Y")
