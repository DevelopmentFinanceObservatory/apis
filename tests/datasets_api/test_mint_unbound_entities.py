"""Mints and previews of members that bind no entity through the collection door, the spec
door, the worker and the preview. Each member's fetch names the places the probe resolved for
it, and a member's `content_version` is the same through every door.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Collection

import httpx
import pytest

import materialise.facets as facets
import materialise.place_scope as place_scope
from datasets_api import kinds
from datasets_api.adapter import build_identity_document
from datasets_api.mint_models import Scope
from datasets_api.revision import SentinelRevisionReader
from datasets_api.shape import resolve_shape
from dc_graph import GraphReadError
from dc_sdmx import SdmxClient
from materialise.graph import Facet
from materialise.place_scope import DefaultPlaceScope, read_default_presence
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry, mint_collection_id
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.materialise._scope_availability import ScopeAvailability

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter
from .test_version_record_writer import RecordingVersionRecordWriter
from .test_worker_e2e import api_and_worker

_BASE_URL = "https://datacommons.one.org/core/api"
_KEN, _TGO, _UGA = "country/KEN", "country/TGO", "country/UGA"

# `_AID` is role-bearing and lives in coverage. `_POPULATION` holds data at two countries and
# a worldwide total. `_WORLD` holds a worldwide total alone.
_AID = "AidDisbursements"
_POPULATION = "Count_Person"
_WORLD = "WorldTotal"
_STAT_VARS = (_AID, _POPULATION, _WORLD)
_SUBJECT_DATA = (
    (_POPULATION, _KEN),
    (_POPULATION, _TGO),
    (_POPULATION, "Earth"),
    (_WORLD, "Earth"),
)

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=1,
    earliest_date="2015",
    latest_date="2015",
)
_COLUMNS = (
    "unit,measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,"
    "facetId\r\n"
)
_CELLS = "USD,GrantEquivalent,NotApplicable,OECD_DAC2A,2015,10,,facet1\r\n"


class _PresenceDown(FakeMintGraphReader):
    """Fails every presence read."""

    def entities_with_data_by_variable(self, variable_dcids, entity_dcids, *, deadline=None):
        raise GraphReadError("the presence read failed")


class _NoCountries(FakeMintGraphReader):
    """Lists no country under Earth, and records each presence read it is asked for."""

    def child_places_batch(self, parent_dcids, child_type, *, deadline=None):
        return {parent: [] for parent in parent_dcids}


def _graph(
    reader: type[FakeMintGraphReader] = FakeMintGraphReader,
    *,
    data: Collection[tuple[str, str]] = _SUBJECT_DATA,
) -> FakeMintGraphReader:
    """The role-bearing StatVar, the subject-only StatVars of `data`, the `(StatVar, place)`
    pairs with observations, and Earth's three countries."""
    subject_arcs = {"typeOf": ("StatisticalVariable",), "measuredProperty": ("count",)}
    return reader(
        arcs={
            _AID: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            },
            **{stat_var: subject_arcs for stat_var, _place in data},
        },
        facets={key: [_FACET] for key in data},
        labels={"OECD_DAC2A": "OECD DAC2a"},
        child_places={("Earth", "Country"): [_KEN, _TGO, _UGA]},
    )


def _coverage() -> FakeCoverageService:
    return FakeCoverageService(
        build_coverage_index(
            [(_AID, "facet1", "country/USA", _KEN, [2015])],
            role_names={_AID: RoleNames(donor="provider", recipient="recipient")},
            facet_metadata={
                "facet1": FacetMetadata(
                    provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
                )
            },
        )
    )


def _sdmx(
    recorded: list[httpx.Request],
    *,
    data: Collection[tuple[str, str]] = _SUBJECT_DATA,
    availability: ScopeAvailability | None = None,
    availability_status: int = 200,
) -> SdmxClient:
    return SdmxClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(
            _mixer(
                recorded,
                data=data,
                availability=availability,
                availability_status=availability_status,
            )
        ),
    )


def _mixer(
    recorded: list[httpx.Request],
    *,
    data: Collection[tuple[str, str]] = _SUBJECT_DATA,
    availability: ScopeAvailability | None = None,
    availability_status: int = 200,
) -> Callable[[httpx.Request], httpx.Response]:
    """Serves one row per subject-only variable at each named place it holds data at, as
    `data` lists them, and one role-bearing row for `_AID`, and records each data read.
    Availability reads answer from `availability`, by default the places `data` lists, or fail
    with `availability_status` other than 200."""
    listed = ScopeAvailability(data=data) if availability is None else availability

    def handler(request: httpx.Request) -> httpx.Response:
        if availability_status != 200 and "/availability/" in request.url.path:
            return httpx.Response(availability_status, json={"code": 14, "message": "down"})
        if (response := listed.respond(request)) is not None:
            return response
        recorded.append(request)
        named = str(request.url.params["c[variableMeasured]"]).split(",")
        places = request.url.params.get("c[observationAbout]")
        prefix = "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,"
        if places is None:
            text = f"{prefix}recipient,{_COLUMNS}" + "".join(
                f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{_KEN},{_CELLS}" for stat_var in named
            )
        else:
            text = f"{prefix}observationAbout,{_COLUMNS}" + "".join(
                f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{place},{_CELLS}"
                for stat_var in named
                for place in places.split(",")
                if (stat_var, place) in data
            )
        return httpx.Response(
            200, text=text, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    return handler


def _filters(recorded: list[httpx.Request]) -> dict[str, dict[str, str]]:
    """The entity filters each StatVar was fetched under."""
    return {
        stat_var: {
            key: value
            for key, value in request.url.params.items()
            if key in ("c[observationAbout]", "c[provider]", "c[recipient]")
        }
        for request in recorded
        for stat_var in str(request.url.params["c[variableMeasured]"]).split(",")
    }


# A subject-only member is fetched at every place it holds data at: Kenya, Togo and Earth's
# total for the population (Uganda holds none), Earth for the worldwide total. Coverage's one
# donor answers the role-bearing member.
_EXPECTED_FILTERS = {
    _AID: {"c[provider]": "country/USA"},
    _POPULATION: {"c[observationAbout]": f"{_KEN},{_TGO},Earth"},
    _WORLD: {"c[observationAbout]": "Earth"},
}
_EXPECTED_ROWS = {_AID: 1, _POPULATION: 3, _WORLD: 1}


def _client(
    recorded: list[httpx.Request],
    *,
    graph: FakeMintGraphReader | None = None,
    version_writer: RecordingVersionRecordWriter | None = None,
    data: Collection[tuple[str, str]] = _SUBJECT_DATA,
    availability: ScopeAvailability | None = None,
    availability_status: int = 200,
):
    store = FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=_graph(data=data) if graph is None else graph,
        version_writer=version_writer,
        sdmx=_sdmx(
            recorded,
            data=data,
            availability=availability,
            availability_status=availability_status,
        ),
        revision_reader=SentinelRevisionReader(),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage(),
    )


def _sidecar(stat_var: str) -> dict:
    """The canonical definition of `stat_var` with no entity and no facet bound."""
    shape = resolve_shape(f"dcid:{stat_var}", graph=_graph())
    document = build_identity_document(shape, scope=Scope())
    assert document["entities"] == {}
    return {
        "schema_version": 1,
        "spec_id": recompute_identity(document).spec_id,
        "identity": document,
    }


def _collection_body(order: tuple[str, ...]) -> dict:
    members = [_sidecar(stat_var) for stat_var in order]
    return {
        "schema_version": 1,
        "input": {
            "kind": "collection_id",
            "collection_id": mint_collection_id([member["spec_id"] for member in members]),
            "members": members,
        },
        "options": {"bundle": False},
    }


def _spec_body(stat_var: str) -> dict:
    sidecar = _sidecar(stat_var)
    return {
        "schema_version": 1,
        "input": {"kind": "spec_id", "spec_id": sidecar["spec_id"], "definition": sidecar},
        "options": {"bundle": False},
    }


def _member_versions(manifest: dict) -> dict[str, str]:
    return {
        member["definition"]["spec_id"]: member["content_version"]
        for member in manifest["collection"]["members"]
    }


def _alone(stat_var: str) -> dict:
    """`stat_var`'s own mint through the spec door, on a fresh deployment."""
    recorded: list[httpx.Request] = []
    resp = _client(recorded).post("/api/datasets", json=_spec_body(stat_var))
    assert resp.status_code == 200, resp.text
    assert _filters(recorded) == {stat_var: _EXPECTED_FILTERS[stat_var]}
    return resp.json()


@pytest.mark.parametrize("stat_var", _STAT_VARS)
def test_a_spec_binding_no_entity_mints_through_the_spec_door(stat_var):
    manifest = _alone(stat_var)

    assert manifest["definition"]["identity"]["entities"] == {}
    assert manifest["coverage"]["rows"] == _EXPECTED_ROWS[stat_var]


@pytest.mark.parametrize("order", list(itertools.permutations(_STAT_VARS)))
def test_a_collection_mints_each_member_binding_no_entity_as_that_member_mints_alone(order):
    recorded: list[httpx.Request] = []

    resp = _client(recorded).post("/api/datasets", json=_collection_body(order))

    assert resp.status_code == 200, resp.text
    assert _filters(recorded) == _EXPECTED_FILTERS
    versions = _member_versions(resp.json())
    assert len(versions) == len(order)
    for stat_var in order:
        alone = _alone(stat_var)
        assert versions[alone["definition"]["spec_id"]] == alone["content_version"]


def _through_the_worker(body: dict, recorded: list[httpx.Request]) -> dict:
    """Submit `body` over budget and return the manifest the worker produced."""
    api_client, _operation_store, forwarded = api_and_worker(
        graph=_graph(), sdmx=lambda: _sdmx(recorded), coverage=_coverage
    )
    submitted = api_client.post("/api/datasets", json=body)
    assert submitted.status_code == 202, submitted.text
    operation_id = submitted.json()["operation"]["operation_id"]
    assert forwarded == [(operation_id, 0)]
    polled = api_client.get(f"/api/datasets/operations/{operation_id}")
    assert polled.status_code == 200, polled.text
    assert polled.json()["operation"]["status"] == "done", polled.text
    return polled.json()["manifest"]


@pytest.mark.parametrize("stat_var", _STAT_VARS)
def test_the_worker_mints_a_spec_binding_no_entity_as_the_inline_door_does(
    stat_var, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(kinds.spec_id, "within_sync_budget", lambda estimate, **_: False)
    recorded: list[httpx.Request] = []

    manifest = _through_the_worker(_spec_body(stat_var), recorded)

    assert _filters(recorded) == {stat_var: _EXPECTED_FILTERS[stat_var]}
    assert manifest["coverage"]["rows"] == _EXPECTED_ROWS[stat_var]
    monkeypatch.undo()
    assert manifest["content_version"] == _alone(stat_var)["content_version"]


@pytest.mark.parametrize("order", [_STAT_VARS, _STAT_VARS[::-1]])
def test_the_worker_mints_a_collection_of_members_binding_no_entity(
    order, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(kinds.collection_id, "within_sync_budget", lambda estimate, **_: False)
    recorded: list[httpx.Request] = []

    manifest = _through_the_worker(_collection_body(order), recorded)

    assert _filters(recorded) == _EXPECTED_FILTERS
    monkeypatch.undo()
    versions = _member_versions(manifest)
    for stat_var in order:
        alone = _alone(stat_var)
        assert versions[alone["definition"]["spec_id"]] == alone["content_version"]


@pytest.mark.parametrize("stat_var", _STAT_VARS)
def test_a_preview_binding_no_entity_reports_the_version_the_mint_reports(stat_var):
    recorded: list[httpx.Request] = []
    body = {**_spec_body(stat_var), "preview": {"limit": 10}}
    del body["options"]

    resp = _client(recorded).post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    assert _filters(recorded) == {stat_var: _EXPECTED_FILTERS[stat_var]}
    (member,) = resp.json()["members"]
    assert member["total_rows"] == _EXPECTED_ROWS[stat_var]
    assert member["content_version"] == _alone(stat_var)["content_version"]


def test_a_preview_of_a_statvar_set_binding_no_entity_previews_each_statvar_as_it_mints():
    recorded: list[httpx.Request] = []
    body = {
        "schema_version": 1,
        "input": {"kind": "statvar_set", "statvars": [f"dcid:{s}" for s in _STAT_VARS]},
        "preview": {"limit": 10},
    }

    resp = _client(recorded).post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    assert _filters(recorded) == _EXPECTED_FILTERS
    assert {member["stat_var"]: member["content_version"] for member in resp.json()["members"]} == {
        f"dcid:{stat_var}": _alone(stat_var)["content_version"] for stat_var in _STAT_VARS
    }


def test_a_statvar_set_of_several_statvars_mints_each_as_its_own_collection_member():
    # A `statvar_set` naming several StatVars registers one spec per StatVar, so each
    # resolves its own places: the worldwide total is fetched at Earth beside the countries.
    recorded: list[httpx.Request] = []
    body = {
        "schema_version": 1,
        "input": {"kind": "statvar_set", "statvars": [f"dcid:{_POPULATION}", f"dcid:{_WORLD}"]},
        "options": {"bundle": False},
    }

    resp = _client(recorded).post("/api/datasets", json=body)

    assert resp.status_code == 200, resp.text
    assert _filters(recorded) == {
        stat_var: _EXPECTED_FILTERS[stat_var] for stat_var in (_POPULATION, _WORLD)
    }
    versions = _member_versions(resp.json())
    for stat_var in (_POPULATION, _WORLD):
        alone = _alone(stat_var)
        assert versions[alone["definition"]["spec_id"]] == alone["content_version"]


def test_a_statvar_set_mints_each_statvar_at_its_own_scope_however_many_it_names():
    # The first StatVar holds data in Kenya and Togo, the second in Uganda and at Earth. Minted
    # together, each is fetched at its own places and mints the rows and `content_version` it
    # mints alone.
    first, second = "Count_Wide_A", "Count_Wide_B"
    data = ((first, _KEN), (first, _TGO), (second, _UGA), (second, "Earth"))

    def mint(*stat_vars: str) -> dict[str, tuple[int, str]]:
        """Each minted spec's rows and `content_version`."""
        body = {
            "schema_version": 1,
            "input": {"kind": "statvar_set", "statvars": [f"dcid:{s}" for s in stat_vars]},
            "options": {"bundle": False},
        }
        resp = _client([], data=data).post("/api/datasets", json=body)
        assert resp.status_code == 200, resp.text
        manifest = resp.json()
        members = manifest["collection"]["members"] if "collection" in manifest else [manifest]
        return {
            member["definition"]["spec_id"]: (member["coverage"]["rows"], member["content_version"])
            for member in members
        }

    together = mint(first, second)

    assert together == {**mint(first), **mint(second)}
    assert sorted(rows for rows, _version in together.values()) == [2, 2]


_BODIES = pytest.mark.parametrize(
    "body",
    [lambda: _spec_body(_POPULATION), lambda: _collection_body((_POPULATION, _WORLD))],
    ids=["spec", "collection"],
)


@_BODIES
def test_a_failed_availability_read_mints_the_countries_with_data_and_says_so(body):
    # A gateway status from availability takes the countries fallback for this request: the
    # population at Kenya and Togo, the worldwide total at Earth, which no country holds.
    recorded: list[httpx.Request] = []

    resp = _client(recorded, availability_status=503).post("/api/datasets", json=body())

    assert resp.status_code == 200, resp.text
    assert _filters(recorded)[_POPULATION] == {"c[observationAbout]": f"{_KEN},{_TGO}"}
    assert any(
        warning["code"] == "coverage_below_scope"
        and "full list of places with data was too large or unavailable" in warning["message"]
        for warning in resp.json()["warnings"]
    )


@_BODIES
def test_a_failed_presence_read_refuses_upstream_error_and_records_no_version(body):
    # Availability fails, so the scope falls back to the countries with data. Read as "no
    # country holds data", a failed presence read would mint Earth's worldwide total in place of
    # every country's data.
    recorded: list[httpx.Request] = []
    versions = RecordingVersionRecordWriter()
    client = _client(
        recorded,
        graph=_graph(_PresenceDown),
        version_writer=versions,
        availability_status=503,
    )

    resp = client.post("/api/datasets", json=body())

    assert resp.status_code == 502, resp.text
    problem = resp.json()
    assert (problem["code"], problem["retryable"]) == ("upstream_error", True)
    assert recorded == []
    assert versions.records == []


@_BODIES
def test_an_empty_countries_read_refuses_upstream_error_and_records_no_version(body):
    # Availability fails, so the scope falls back to the countries with data. The graph places
    # every country under Earth, so a read naming none failed. Read as "no country", it would
    # mint Earth's worldwide total in place of every country's data.
    recorded: list[httpx.Request] = []
    versions = RecordingVersionRecordWriter()
    graph = _graph(_NoCountries)
    client = _client(recorded, graph=graph, version_writer=versions, availability_status=503)

    resp = client.post("/api/datasets", json=body())

    assert resp.status_code == 502, resp.text
    problem = resp.json()
    assert (problem["code"], problem["retryable"]) == ("upstream_error", True)
    assert graph.presence_reads == []
    assert recorded == []
    assert versions.records == []


_EVERY_PLACE = {_POPULATION: (_KEN, _TGO, "Earth"), _WORLD: ("Earth",)}
# The fallback: the population's countries, and Earth for the worldwide total, which no country
# holds.
_COUNTRIES = {_POPULATION: (_KEN, _TGO), _WORLD: ("Earth",)}


@pytest.mark.parametrize("full_list", ["answers", "capped"])
@pytest.mark.parametrize("stat_var", [_POPULATION, _WORLD])
def test_a_mint_and_a_preview_binding_no_place_serve_the_default_scope_search_resolves(
    stat_var, full_list, monkeypatch: pytest.MonkeyPatch
):
    if full_list == "capped":
        # One place: the population's three are past it, the worldwide total's one is not.
        monkeypatch.setattr(place_scope, "MAX_EVERY_PLACE", 1)
    every_place = full_list == "answers" or stat_var == _WORLD
    # Search takes the default place scope from the same rule over the same mixer.
    scope = read_default_presence(
        graph=_graph(), availability=ScopeAvailability(data=_SUBJECT_DATA), stat_vars=[stat_var]
    ).scope(stat_var)
    minted: list[httpx.Request] = []
    previewed: list[httpx.Request] = []
    preview_body = {**_spec_body(stat_var), "preview": {"limit": 10}}
    del preview_body["options"]

    mint = _client(minted).post("/api/datasets", json=_spec_body(stat_var))
    preview = _client(previewed).post("/api/datasets", json=preview_body)

    assert mint.status_code == 200, mint.text
    assert preview.status_code == 200, preview.text
    assert scope == DefaultPlaceScope(
        places=(_EVERY_PLACE if every_place else _COUNTRIES)[stat_var], every_place=every_place
    )
    for recorded in (minted, previewed):
        assert _filters(recorded)[stat_var]["c[observationAbout]"].split(",") == list(scope.places)
    (member,) = preview.json()["members"]
    about = member["sample"]["columns"].index("observationAbout")
    assert sorted({row[about] for row in member["sample"]["rows"]}) == sorted(scope.places)
    # The mint warns of the countries fallback exactly when the scope is one.
    countries_only = [
        warning
        for warning in mint.json()["warnings"]
        if warning["code"] == "coverage_below_scope"
        and "full list of places with data was too large or unavailable" in warning["message"]
    ]
    assert bool(countries_only) is not every_place


_GHOST, _ZAF = "country/SouthAfrica", "country/ZAF"


class _GhostGraph(FakeMintGraphReader):
    """Holds no node for `_GHOST`, though observations are recorded there, as the development
    mixer keys some data to it beside `_ZAF`."""

    def node_arcs_many(self, dcids, *, properties, individual_probe_budget=None, deadline=None):
        arcs = super().node_arcs_many(dcids, properties=properties, deadline=deadline)
        return {dcid: None if dcid == _GHOST else node for dcid, node in arcs.items()}


def test_a_place_the_graph_holds_no_node_for_is_neither_stated_nor_fetched():
    # Availability lists the population at `_GHOST` and `_ZAF`; the graph knows only `_ZAF`.
    data = ((_POPULATION, _GHOST), (_POPULATION, _ZAF))
    minted: list[httpx.Request] = []

    resp = _client(minted, graph=_graph(_GhostGraph, data=data), data=data).post(
        "/api/datasets", json=_spec_body(_POPULATION)
    )

    assert resp.status_code == 200, resp.text
    assert _filters(minted)[_POPULATION] == {"c[observationAbout]": _ZAF}
    assert [warning["code"] for warning in resp.json()["warnings"]].count(
        "coverage_below_scope"
    ) == 0


def test_a_scope_the_request_deadline_leaves_unread_refuses_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    def out_of_time(*, stat_vars, **_kwargs):
        raise place_scope.PlaceScopeDeadlineError(list(stat_vars))

    monkeypatch.setattr(facets, "read_default_presence", out_of_time)
    recorded: list[httpx.Request] = []

    resp = _client(recorded).post("/api/datasets", json=_spec_body(_POPULATION))

    assert resp.status_code == 504, resp.text
    problem = resp.json()
    assert (problem["code"], problem["retryable"]) == ("timeout", True)
    assert recorded == []
