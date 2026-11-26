"""Search's answer and the mint of the same request state the same held-back runs
(`search.engine.retrieval.held_back`, `datasets_api.manifest.held_back_coverage`), whatever
the window.

The per-series rows read the sources of `tests.materialise.test_held_back`: a CRS-like source
marked partial for 2025 whose ninety recipients end in 2024 or reach 2025, and an IDS-like
source whose three debtors hold reported values to 2025 and `Estimated` projections from 2026
to 2030, for a StatVar no coverage index holds. The mint holds back the partial and projected
rows unless its window opts in, so its frame holds no row of a run it states held back.

The role-bearing rows read IDS debt flows per pair and in bulk, both sides from the coverage
probe a mint reads. The composite row reads the CRS inclusive composite, whose CRS component
reaches a partial year, and the groupby row sums a frame whose partial year the mint cut."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from dataclasses import replace
from functools import partial

import httpx
import pytest
from pydantic import TypeAdapter

from code_index.shared import get_index
from datasets_api.adapter import identity_requested_window
from datasets_api.config import DatasetsBuildSettings
from datasets_api.manifest import _window_holds, held_back_coverage
from materialise import EnrichmentLookup, MaterialisationRequest, materialise_spec
from materialise.graph import Facet
from materialise.held_runs import HeldRun
from materialise.window import RequestedWindow
from origin_catalogue import Catalogue, bound_catalogue, current_catalogue
from search.engine.extract import DateRequest
from search.engine.region.composite_answer import (
    ComponentReads,
    CompositeRelaunch,
    settled_composite,
)
from search.engine.region.result import RegionResult
from search.engine.retrieval.facet_policy import NO_DECLARED_POLICIES
from search.engine.retrieval.held_back import held_back, held_runs, window_held_back
from search.engine.retrieval.read_plan import (
    BulkPlan,
    PlanSeries,
    PlanYears,
    kept_dates,
    kept_points,
    plan_pairs,
)
from search.engine.spec_build import build_spec
from search.models import BaseIdentity, IdentityWindow, TimeWindow
from search.scope.composite import CompositeMint, served_components
from search.scope.tables import ScopeFacts
from series_coverage.index import CoverageIndex, served_until
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from tests.datasets_api import test_composite_download as composite_download
from tests.datasets_api import test_serve_groupby_collapse as collapse
from tests.datasets_api._artifact_doubles import FakeArtifactStore
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.datasets_api._version_record_store import FakeVersionRecordDatasetStore
from tests.datasets_api.test_definition_door import _client
from tests.engine import test_facet_election_agreement as agreement
from tests.engine import test_role_bearing_pipeline_e2e as role_bearing
from tests.engine._assemble_fixtures import _make_coverage, _make_spec
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph
from tests.materialise import _weo_recorded as weo
from tests.materialise import test_held_back as sources
from tests.materialise.conftest import FakeGraphReader
from tests.origin_catalogue.fake_catalogue import dev_provenance_nodes, documenting
from tests.scope import test_basis as basis

type _Stated = tuple[str, bool, str, int, int]


@pytest.fixture(autouse=True)
def _sdmx_and_catalogue(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """SDMX serves the periods a read asks for, and CRS marks 2025 partial as its dev graph
    node does."""
    monkeypatch.setattr(weo, "sdmx", partial(weo.sdmx, honour_periods=True))
    unmarked = {"lastDataRefreshDate": ("2026-10-29",)}
    with bound_catalogue(
        documenting({sources._CRS: dev_provenance_nodes()[sources._CRS], sources._IDS: unmarked})
    ):
        yield


def _searched(source: sources._Source, years: PlanYears) -> list[_Stated]:
    places, facets, _values = source
    plan = plan_pairs(
        [
            PlanSeries(
                stat_var=weo.GDP,
                role_tuple=(("observationAbout", place),),
                candidates=tuple(facets[place]),
            )
            for place in places
        ],
        members=[weo.GDP],
        years=years,
        declared=NO_DECLARED_POLICIES,
    )
    return [
        (h.kind, h.served, h.source, h.start_year, h.end_year)
        for h in held_back(plan, held_runs(plan, coverage_index=None))
    ]


def _minted(source: sources._Source, window: RequestedWindow) -> list[_Stated]:
    result = sources._mint(source, window)
    request = weo._request(source[0], window=window, facet_bindings=None)
    return [
        (h["kind"], h["served"], h["source"], h["start_year"], h["end_year"])
        for h in held_back_coverage(result, requests=[request])
    ]


_PARTIAL = sources._crs(ending_2024=80, reaching_2025=10)
_RAGGED_PROJECTIONS = sources._source(
    {
        "country/D0": [
            (sources._IDS, None, range(2015, 2023)),
            (sources._IDS, sources._ESTIMATED, range(2023, 2031)),
        ],
        "country/D1": [
            (sources._IDS, None, range(2015, 2025)),
            (sources._IDS, sources._ESTIMATED, range(2025, 2031)),
        ],
    }
)
"""Two debtors whose reported values end in 2022 and 2024, each projected from the year
after, so the source's projected run starts in 2023."""


@pytest.mark.parametrize(
    ("source", "run"),
    [
        pytest.param(_PARTIAL, ("partial", sources._CRS, 2025, 2025), id="partial"),
        pytest.param(sources._IDS_DEBTORS, ("projected", sources._IDS, 2026, 2030), id="projected"),
        pytest.param(
            _RAGGED_PROJECTIONS,
            ("projected", sources._IDS, 2023, 2030),
            id="projected-from-the-earliest-end",
        ),
    ],
)
@pytest.mark.parametrize(
    ("years", "window", "served"),
    [
        pytest.param(
            TimeWindow(start_year=2016, end_year=2018),
            RequestedWindow(kind="range", start_year=2016, end_year=2018),
            False,
            id="2016-2018",
        ),
        pytest.param("latest", RequestedWindow(kind="latest"), False, id="latest"),
        pytest.param(None, RequestedWindow(kind="all"), False, id="every-year"),
        pytest.param(
            TimeWindow(start_year=2015, end_year=2030, include=["partial", "projected"]),
            RequestedWindow(
                kind="range",
                start_year=2015,
                end_year=2030,
                include=frozenset({"partial", "projected"}),
            ),
            True,
            id="opted-in",
        ),
    ],
)
def test_search_and_the_mint_of_one_request_state_the_same_runs(
    source: sources._Source,
    run: tuple[str, str, int, int],
    years: PlanYears,
    window: RequestedWindow,
    served: bool,
) -> None:
    kind, provenance, start, end = run

    stated = [(kind, served, provenance, start, end)]

    assert _searched(source, years) == stated
    assert _minted(source, window) == stated


def _role_metadata(estimate_period: str | None) -> dict[str, FacetMetadata]:
    """IDS's recorded debt flows, annual, and its `Estimated` projections, whose coverage
    states `estimate_period`."""
    return {
        "f-rec": FacetMetadata(provenance_id=sources._IDS, observation_period="P1Y"),
        "f-est": FacetMetadata(
            provenance_id=sources._IDS,
            measurement_method="Estimated",
            observation_period=estimate_period,
        ),
    }


_ROLE_FACETS = {
    agreement._KEN: [("f-rec", range(2010, 2026), None), ("f-est", range(2026, 2031), "Estimated")],
    agreement._NOR: [("f-rec", range(2010, 2024), None)],
}
_ROLE_PROVENANCES = {"f-rec": sources._IDS, "f-est": sources._IDS}
_ROLE_ROWS = [
    agreement._role_row(debtor, facet_id, years)
    for debtor, facets in _ROLE_FACETS.items()
    for facet_id, years, _method in facets
]
_DEBTORS = {role_bearing._SV_DCID: RoleNames(donor=None, recipient="debtor")}


def _role_answer(
    monkeypatch: pytest.MonkeyPatch, metadata: dict[str, FacetMetadata]
) -> tuple[RegionResult, CoverageIndex]:
    """Search's answer for Kenya's and Norway's debt flows from 2015 to 2020, and the
    coverage index it read."""
    index = patch_coverage(monkeypatch, _ROLE_ROWS, role_names=_DEBTORS, facet_metadata=metadata)
    result = role_bearing._run(
        entities=["Kenya", "Norway"],
        sv_dcid=role_bearing._SV_DCID,
        sv_arcs=role_bearing._declaring_arcs(observation_properties=["debtor"]),
        resolve={"Kenya": agreement._KEN, "Norway": agreement._NOR},
        sv_score=0.9,
        role_query="debt flows to Kenya and Norway",
        sdmx=FakeSdmxClient(observations={}, availability={}),
        date_request=DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False),
        coverage=FakeCoverageService(index).snapshot(),
    )
    assert result.status == "definite", result
    return result, index


def _role_searched(
    monkeypatch: pytest.MonkeyPatch, metadata: dict[str, FacetMetadata], *, max_pairs: int
) -> list[_Stated]:
    """Search's runs for Kenya's and Norway's debt flows from 2015 to 2020, per pair or in
    bulk."""
    result, _index = _role_answer(monkeypatch, metadata)
    return [
        (h.kind, h.served, h.source, h.start_year, h.end_year)
        for h in result.specs[0].resolved.coverage.held_back or []
    ]


def _role_minted(metadata: dict[str, FacetMetadata]) -> list[_Stated]:
    """The mint's runs for the same flows and years, over the same coverage."""
    index = build_coverage_index(_ROLE_ROWS, role_names=_DEBTORS, facet_metadata=metadata)
    request = MaterialisationRequest(
        stat_var=f"dcid:{role_bearing._SV_DCID}",
        role_bindings={"debtor": [agreement._KEN, agreement._NOR]},
        declared_roles=("debtor",),
        facet_bindings={},
        window=RequestedWindow(kind="range", start_year=2015, end_year=2020),
    )
    reader = FakeGraphReader()
    result = materialise_spec(
        request,
        sdmx=agreement._role_materialise_sdmx(_ROLE_FACETS, _ROLE_PROVENANCES),
        graph=reader,
        enrichment=EnrichmentLookup(reader),
        parties={},
        materialiser_build="build-test",
        capability_version="cap-test",
        coverage=index,
    )
    return [
        (h["kind"], h["served"], h["source"], h["start_year"], h["end_year"])
        for h in held_back_coverage(result, requests=[request])
    ]


@pytest.mark.parametrize("max_pairs", [120, 1], ids=["per-pair", "bulk"])
@pytest.mark.parametrize(
    ("estimate_period", "stated"),
    [
        pytest.param("P1Y", [("projected", False, sources._IDS, 2026, 2030)], id="annual"),
        # An estimate whose coverage states no period is a cadence of its own
        # (`materialise.facets.cadences_by_member`), so it continues no annual series, though
        # the coverage index holds its years past the recorded ones.
        pytest.param(None, [], id="no-stated-period"),
    ],
)
def test_a_role_bearing_answer_and_its_mint_state_the_same_runs(
    monkeypatch: pytest.MonkeyPatch,
    search_settings: Callable[..., object],
    max_pairs: int,
    estimate_period: str | None,
    stated: list[_Stated],
) -> None:
    search_settings(obs_preview_max_pairs=max_pairs)
    metadata = _role_metadata(estimate_period)

    assert _role_searched(monkeypatch, metadata, max_pairs=max_pairs) == stated
    assert _role_minted(metadata) == stated


def test_a_bulk_answer_s_runs_read_the_coverage_candidates_alone(
    monkeypatch: pytest.MonkeyPatch, search_settings: Callable[..., object]
) -> None:
    """A bulk plan's runs read each member's coverage candidates and none of the admitted
    pairs a mint's probe also builds, a scan of every coverage row of the member, so a
    plan read again after a narrowing pays no scan."""
    search_settings(obs_preview_max_pairs=1)
    result, index = _role_answer(monkeypatch, _role_metadata("P1Y"))
    (plan,) = result.served_plans
    assert isinstance(plan, BulkPlan)

    def scanned(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the runs read the admitted pairs")

    monkeypatch.setattr(CoverageIndex, "admitted_pairs", scanned)

    assert held_runs(replace(plan, held_runs=None), coverage_index=index) == (
        HeldRun(kind="projected", source=sources._IDS, start_year=2026, end_year=2030),
    )


_OPEN_FROM_2020 = {"kind": "range", "start_year": 2020, "include": ["projected"]}


@pytest.mark.parametrize(
    ("window", "start", "served"),
    [
        pytest.param(_OPEN_FROM_2020, 2027, False, id="open-end-before-the-run"),
        pytest.param(_OPEN_FROM_2020, 2025, True, id="open-end-in-the-run"),
        pytest.param({**_OPEN_FROM_2020, "end_year": 2030}, 2027, True, id="named-end-past-clock"),
        pytest.param(
            {"kind": "range", "start_year": 2020, "end_year": 2030}, 2027, False, id="no-opt-in"
        ),
    ],
)
def test_a_composite_window_serves_a_run_as_the_mint_of_its_identity_does(
    window: dict, start: int, served: bool
) -> None:
    """A composite's runs are served as its identity's window serves them, and a mint of the
    identity ends an open window at the clock year (2026 here) before it tests a run's
    years, so Search does too."""
    run = HeldRun(kind="projected", source=sources._IDS, start_year=start, end_year=2030)
    minted = identity_requested_window({"requested_window": window})

    (searched,) = window_held_back(TypeAdapter(IdentityWindow).validate_python(window), [run])

    assert searched.served is served
    assert (run.kind in minted.include and _window_holds(minted, run)) is served


def _composite_identity() -> dict:
    """The CRS inclusive education composite over 2019 to 2024."""
    identity = composite_download._identity()
    identity["requested_window"] = {"kind": "range", "start_year": 2019, "end_year": 2024}
    return identity


_CRS_OTHER_FACET = "facet_crs_commitment"

type _CompositeRow = tuple[str, str, str, tuple[int, ...]]
"""One member's series under one facet: its dcid, facet id, measurement method and years."""


def _composite_rows(*, unelected: bool) -> list[_CompositeRow]:
    """The CRS members to 2025, a year CRS marks partial, and the imputed member to 2024, all
    grant equivalents. With `unelected`, the CRS members' grant equivalents end in 2024 and
    each holds a CRS commitments facet in 2024 and 2025 that the component's election leaves
    out, as it ranks the grant equivalents' six years in the window above it."""
    members = composite_download._members()
    bilateral = [member.removeprefix("dcid:") for member in members["bilateral"]]
    imputed = members["imputed"][0].removeprefix("dcid:")
    crs_years = tuple(range(2019, 2025 if unelected else 2026))
    return [
        *(
            (dcid, composite_download._CRS_FACET, "GrantEquivalent", crs_years)
            for dcid in bilateral
        ),
        *((dcid, _CRS_OTHER_FACET, "Commitment", (2024, 2025)) for dcid in bilateral if unelected),
        (imputed, composite_download._IMPUTED_FACET, "GrantEquivalent", tuple(range(2020, 2025))),
    ]


def _composite_coverage(rows: list[_CompositeRow]) -> FakeCoverageService:
    """The coverage index of `rows`, each facet described by its source and method, in USD."""
    provenance = {**composite_download._SDMX_PROVENANCE, _CRS_OTHER_FACET: sources._CRS}
    methods = {facet: method for _dcid, facet, method, _years in rows}
    index = build_coverage_index(
        [
            (dcid, facet, composite_download._PROVIDER, composite_download._RECIPIENT, list(years))
            for dcid, facet, _method, years in rows
        ],
        role_names={
            dcid: RoleNames(donor="financingProvider", recipient="financingRecipient")
            for dcid, _facet, _method, _years in rows
        },
        facet_metadata={
            facet: FacetMetadata(
                provenance_id=provenance[facet], measurement_method=method, unit="USD"
            )
            for facet, method in methods.items()
        },
    )
    return FakeCoverageService(index)


def _composite_sdmx(rows: list[_CompositeRow], request: httpx.Request) -> httpx.Response:
    """The rows of `rows` for the members `request` names, as SDMX CSV."""
    provenance = {**composite_download._SDMX_PROVENANCE, _CRS_OTHER_FACET: sources._CRS}
    named = str(request.url.params.get("c[variableMeasured]", "")).split(",")
    body = composite_download._CSV_HEADER + "".join(
        f"dataflow,DC:DF_OBS(1.0.0),I,{dcid},{composite_download._PROVIDER},"
        f"{composite_download._RECIPIENT},USD,{method},NotApplicable,"
        f"{provenance[facet]},{year},100,,{facet}\r\n"
        for dcid, facet, method, years in rows
        if dcid in named
        for year in years
    )
    return httpx.Response(
        200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
    )


def _composite_searched(
    identity: dict, coverage: FakeCoverageService, *, path: str
) -> list[_Stated]:
    """The runs Search's composite answer states on `path` ("plain", "streamed" or
    "embedded") once its components' read plans settle (`composite_answer.settled_composite`):
    each component resolved as a request reading no preview resolves it, over a graph holding
    every member and both places. A streamed answer states them as its `response` event
    carries it, before its preview task runs."""
    facts = ScopeFacts(index=get_index(), coverage=coverage.snapshot())
    mint = CompositeMint(
        identity=BaseIdentity.model_validate(identity),
        spec_id="spec_test",
        served=tuple(served_components(identity, facts=facts)),
        facts=facts,
    )
    spec = _make_spec(sv_dcid="Amount_Aid", purpose_dcid="DAC/Education", place_dcid="Kenya")
    answered = build_spec(
        composite=mint,
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=spec.resolved.shape,
        slots=[],
        stat_vars=[],
        entities=[],
        coverage=_make_coverage(),
    )
    members = sorted({member for item in mint.served for member in item.members or ()})
    graph = FakeGraph(
        nodes={
            composite_download._PROVIDER: {"label": "France", "type": "Country"},
            composite_download._RECIPIENT: {"label": "Kenya", "type": "Country"},
            **{dcid: basis._member_node(dcid) for dcid in members},
        }
    )
    reads = ComponentReads(
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=coverage.snapshot(),
        materialise_graph=None,
        materialise_sdmx=None,
        rate_cache=None,
        deadline=None,
        catalogue=current_catalogue(),
    )
    region = RegionResult(
        variable_text=None,
        status="definite",
        specs=(answered,),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
        observations_relaunch=CompositeRelaunch(spec=answered, reads=reads, served=mint.served),
    )
    settled = asyncio.run(
        settled_composite(
            region,
            stream=path == "streamed",
            embed=path == "embedded",
            plans_semaphore=None,
            preview_semaphore=None,
        )
    )
    assert (settled.observations_relaunch is not None) == (path == "streamed")
    return [
        (h.kind, h.served, h.source, h.start_year, h.end_year)
        for h in settled.spec.resolved.coverage.held_back or []
    ]


@pytest.mark.parametrize(
    ("unelected", "stated"),
    [
        pytest.param(False, [("partial", False, sources._CRS, 2025, 2025)], id="elected-facet"),
        # The commitments facet reaching 2025 is a candidate of every CRS member and no member's
        # elected facet, so neither side states its partial year.
        pytest.param(True, [], id="unelected-facet"),
    ],
)
def test_a_composite_answer_and_its_mint_state_the_same_runs(
    serve_catalogue: Callable[[Catalogue], None], unelected: bool, stated: list[_Stated]
) -> None:
    """The CRS component's rows reach 2025, which CRS marks partial, so the composite's mint
    states the run, merged across its components, and Search's composite answer states the
    same, unserved, as the window names no opt-in, once its components' read plans settle,
    whether the answer reads no preview, streams it or embeds it. Where only a facet the
    component's election leaves out reaches 2025, neither states it."""
    identity, rows = _composite_identity(), _composite_rows(unelected=unelected)
    coverage = _composite_coverage(rows)
    serve_catalogue(current_catalogue())
    client = _client(
        graph=FakeMintGraphReader(
            labels={composite_download._PROVIDER: "France", composite_download._RECIPIENT: "Kenya"}
        ),
        sdmx_transport=httpx.MockTransport(lambda request: _composite_sdmx(rows, request)),
        coverage=coverage,
        artifact_store=FakeArtifactStore(),
        settings=DatasetsBuildSettings.from_env(),
    )

    minted = composite_download._mint(client, identity=identity)

    assert minted.status_code == 200, minted.text
    assert [
        (h["kind"], h["served"], h["source"], h["start_year"], h["end_year"])
        for h in minted.json()["coverage"].get("held_back", [])
    ] == stated
    assert {
        path: _composite_searched(identity, coverage, path=path)
        for path in ("plain", "streamed", "embedded")
    } == dict.fromkeys(("plain", "streamed", "embedded"), stated)


@pytest.mark.parametrize(
    ("include", "served"),
    [
        pytest.param(["partial"], {2025}, id="opted-in"),
        pytest.param(None, set(), id="default"),
    ],
)
def test_a_composite_component_serves_the_partial_year_its_window_opts_in_to(
    include: list[str] | None, served: set[int]
) -> None:
    """Both components' rows reach 2025, which CRS marks partial and the imputed source does
    not. Under a window naming 2025 that opts in to partial years, as a mint of the
    composite reads its components, the CRS component holds 2025, so the stacked composite
    serves and previews it in both. Under the default window the CRS component holds none
    of it, so neither serves it."""
    identity = _composite_identity()
    identity["requested_window"] = {
        "kind": "range",
        "start_year": 2019,
        "end_year": 2025,
        **({"include": include} if include else {}),
    }
    rows = [
        (
            dcid,
            facet,
            method,
            (*years, 2025) if facet == composite_download._IMPUTED_FACET else years,
        )
        for dcid, facet, method, years in _composite_rows(unelected=False)
    ]
    facts = ScopeFacts(index=get_index(), coverage=_composite_coverage(rows).snapshot())

    found = {
        item.component.role: (item.served & {2025}, item.preview_years & {2025})
        for item in served_components(identity, facts=facts)
    }

    assert found == {"bilateral": (served, served), "imputed": (served, served)}


def _two_year_sdmx() -> httpx.MockTransport:
    """Both of `test_serve_groupby_collapse`'s members at Kenya in 2019 and 2020."""
    body = collapse._CSV_HEADER + "".join(
        f"dataflow,DC:DF_OBS(1.0.0),I,{member},{collapse._RECIPIENT},USD,GrantEquivalent,"
        f"NotApplicable,{collapse._PROVENANCE},{year},{value},,{collapse._FACET_ID}\r\n"
        for member, value in ((collapse._MEMBER_A, 1000), (collapse._MEMBER_B, 500))
        for year in (2019, 2020)
    )
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )
    )


def test_a_groupby_sum_states_the_runs_of_the_frame_it_sums(
    serve_catalogue: Callable[[Catalogue], None],
) -> None:
    """The source marks 2020 partial and the window names no opt-in, so the mint cuts 2020
    and states the run held back. The sum of both members over the recipient states the same
    run, unserved, as its base does, minted and answered from its record."""
    collapse._marking(serve_catalogue, "2020")
    client = _client(
        graph=collapse._graph(),
        sdmx_transport=_two_year_sdmx(),
        version_store=FakeVersionRecordDatasetStore(),
        code_index=collapse._code_index(),
        coverage=FakeCoverageService(
            build_coverage_index(
                [
                    (member, collapse._FACET_ID, ABSENT_ENTITY, collapse._RECIPIENT, [2019, 2020])
                    for member in (collapse._MEMBER_A, collapse._MEMBER_B)
                ],
                role_names={
                    member: RoleNames(donor=None, recipient="recipient")
                    for member in (collapse._MEMBER_A, collapse._MEMBER_B)
                },
                facet_metadata={
                    collapse._FACET_ID: FacetMetadata(
                        provenance_id=collapse._PROVENANCE,
                        measurement_method=collapse._MEASUREMENT_METHOD,
                        unit=collapse._UNIT,
                    )
                },
            )
        ),
    )
    spec_id, sidecar = collapse._spec_id_and_sidecar()

    def stated(options: dict) -> list[_Stated]:
        minted = collapse._mint(client, spec_id=spec_id, sidecar=sidecar, options=options)
        window = minted["coverage"]["window"]
        assert (window["start_year"], window["end_year"]) == (2019, 2019)
        return [
            (h["kind"], h["served"], h["source"], h["start_year"], h["end_year"])
            for h in minted["coverage"].get("held_back", [])
        ]

    run = [("partial", False, collapse._PROVENANCE, 2020, 2020)]
    assert stated({"bundle": False}) == run
    summed = {"bundle": False, "groupby": {"by": ["recipient"], "op": "sum"}}
    assert stated(summed) == run
    # A repeat answers from the recorded version.
    assert stated(summed) == run


@pytest.mark.parametrize(
    ("first", "last", "served"),
    [
        pytest.param("2023-01", "2025-09", "2025-05", id="monthly"),
        pytest.param("2023", "2025", "2024", id="annual"),
    ],
)
def test_search_cuts_a_facet_where_the_mint_cuts_its_rows_under_a_sub_annual_mark(
    first: str, last: str, served: str
) -> None:
    """The source marks June 2025 partial. A range mint keeps a monthly series' months to
    May 2025 and none of an annual series' 2025, and Search's cut of the facet's newest
    period (`served_until`) ends where the frame does. Search's dates of the facet listing
    every period (`read_plan.kept_dates`, `read_plan.kept_points`) are the frame's."""
    periods = (
        sources._months(first, last) if len(first) == 7 else [str(y) for y in range(2023, 2026)]
    )
    facet = Facet(
        facet_id="f",
        provenance_id=sources._MONTHLY,
        measurement_method=None,
        unit=sources._USD,
        obs_count=len(periods),
        earliest_date=first,
        latest_date=last,
    )
    place = "country/M0"
    window = RequestedWindow(kind="range", start_year=2023, end_year=2026)
    marked = {"lastDataRefreshDate": ("2026-10-29",), "partialObservationDate": ("2025-06",)}

    with bound_catalogue(documenting({sources._MONTHLY: marked})):
        minted = sources._mint(
            ([place], {place: [facet]}, {(place, "f"): dict.fromkeys(periods, 1.0)}), window
        )
        through = served_until(source=sources._MONTHLY, method=None, newest=last)
        dated = replace(facet, dates=tuple(periods), values=(1.0,) * len(periods))
        kept_span = kept_dates(dated, (first, last))
        kept = kept_points(dated, periods, include=frozenset())

    framed = sources._years(minted)
    assert max(framed) == served
    assert through == served
    assert kept_span == (min(framed), max(framed))
    assert set(kept) == framed
