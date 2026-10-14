"""Relations between the doors on a UN System answer, and what Search serves of a superseded
provenance or an excluded series.

The first half runs Search, the resubmit and the Datasets mint through real graph clients
over one wire double of our mixer and the UN deployment (`tests.materialise.un_deployment`),
so each test reads the requests the doors sent. The second half runs the pipeline over the
fake graph with a recall that returns a variable held only under `dc/base/UN_SDG` and a
variable of an excluded UN series beside the ones the catalogue admits.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from dc_sdmx import SdmxClient
from origin_catalogue import BASE, UN_SYSTEM, CatalogueSnapshot
from origin_catalogue.graph import OriginGraphClient, build_graph_client
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.region import resolve_spec_resubmit
from search.engine.region.pipeline import resolve_variable
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions
from search.engine.retrieval.derive import derive_shapes
from search.models import ResubmitIdentityPayload, Spec
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from statvar_members import shared_enumerator
from tests.datasets_api._app_factories import make_client
from tests.datasets_api._artifact_doubles import FakeArtifactStore
from tests.datasets_api._snapshot_doubles import FakeSnapshotReader
from tests.datasets_api.conftest import FakeDatasetStore
from tests.datasets_api.test_mint_unbound_entities import _coverage
from tests.datasets_api.test_pin_door import FakeArtifactReferenceWriter
from tests.datasets_api.test_un_doors import _sidecar
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine.test_recall_transfer import (
    _FACET,
    _KEN,
    _OURS,
    _QUERY,
    _SDG,
    _SDG_ROW,
    _SDG_SERIES,
    _WHO_ROW,
    _elected,
    _ElectFirstLLM,
    _names,
    _resolve,
    _sv_node,
    _topic_node,
)
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise.test_observation_rows import _sdmx_never_called
from tests.materialise.un_deployment import (
    OURS_API,
    UN_API,
    UN_SECOND_VARIABLE,
    UN_UNIT,
    UN_VARIABLE,
    UnDeployments,
)
from tests.origin_catalogue.fake_catalogue import (
    fake_catalogue_service,
    provenance_row,
    restrictive_catalogue,
    un_mirror,
    un_variable,
)

_OURS_HOST = httpx.URL(OURS_API).host
_UN_HOST = httpx.URL(UN_API).host

_WDI = "dc/base/WorldDevelopmentIndicators"
_BASE_SDG_PROVENANCE = "dc/base/UN_SDG"
_BASE_SDG = "sdg/SH_DYN_MORT"

_WHO_SERIES = "undata/who/SERIES-MORT_U5_RATE"
_WEALTH = "undataWealthQuantile"

_PLACES = FakeGraph(nodes={_KEN: {"label": "Kenya", "type": "Country"}}, resolve={"Kenya": _KEN})
"""The nodes our mixer holds on the wire, and the graph the entity resolver reads."""


def _who_catalogue() -> CatalogueSnapshot:
    """The UN tables hold the WHO series the UN deployment double serves, two variables that
    differ by wealth quintile. `_BASE_SDG`, which our resolver recalls, is superseded and
    stands as the first of them."""

    def quintile(code: str) -> dict[str, tuple[str, str, str]]:
        return {_WEALTH: ("Wealth quintile", f"undata/WEALTH_QUANTILE-{code}", code)}

    return restrictive_catalogue(
        provenances=[
            provenance_row(_SDG_ROW, UN_SYSTEM, segment="sdg"),
            provenance_row(_WHO_ROW, UN_SYSTEM, segment="who"),
        ],
        withheld_variables=[_BASE_SDG],
        successors={_BASE_SDG: UN_VARIABLE},
        un_mirror=un_mirror(
            {
                UN_VARIABLE: un_variable(
                    _WHO_SERIES, name="Under-five mortality [Q1]", constraints=quintile("Q1")
                ),
                UN_SECOND_VARIABLE: un_variable(
                    _WHO_SERIES, name="Under-five mortality [Q2]", constraints=quintile("Q2")
                ),
            }
        ),
    )


def _our_nodes(request: httpx.Request) -> httpx.Response:
    """Our mixer's node read, answered from `_PLACES` and the schema nodes beneath it."""
    payload = json.loads(request.content)
    prop = payload["property"]
    data: dict = {}
    for dcid in payload["nodes"]:
        arcs = dict(_PLACES.node_arcs(dcid) or {})
        if type_of := _PLACES.node_type(dcid):
            arcs.setdefault("typeOf", {"nodes": [{"dcid": type_of}]})
        if label := _PLACES.node_label(dcid):
            arcs.setdefault("name", {"nodes": [{"value": label}]})
        if not prop.startswith("->"):
            arcs = {}
        elif prop != "->*":
            wanted = prop[2:].strip("[]").replace(" ", "").split(",")
            arcs = {name: arc for name, arc in arcs.items() if name in wanted}
        data[dcid] = {"arcs": arcs} if arcs else {}
    return httpx.Response(200, json={"data": data})


def _resolved(scored: Mapping[str, float]) -> httpx.Response:
    candidates = [
        {"dcid": dcid, "metadata": {"score": score}, "typeOf": ["StatisticalVariable"]}
        for dcid, score in scored.items()
    ]
    return httpx.Response(200, json={"entities": [{"candidates": candidates}]})


@dataclass
class _Wire:
    """Our mixer and the UN deployment behind every `httpx.Client`. The UN's indicator resolve
    returns `un_recall` and ours `our_recall`. Our observation read holds one facet for each
    `(variable, place)` in `our_facets`, under the provenance it names, whatever domain
    filter the read carries. Every other read is `UnDeployments`'."""

    deployments: UnDeployments = field(default_factory=UnDeployments)
    un_recall: dict[str, float] = field(default_factory=lambda: {UN_VARIABLE: 0.9})
    our_recall: dict[str, float] = field(default_factory=lambda: {_BASE_SDG: 0.9})
    our_facets: dict[tuple[str, str], str] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        ours = request.url.host == _OURS_HOST
        if request.url.path.endswith("/v2/resolve"):
            return _resolved(self.our_recall if ours else self.un_recall)
        if ours and request.url.path.endswith("/v2/node"):
            return _our_nodes(request)
        if ours and request.url.path.endswith("/v2/observation") and self.our_facets:
            return self._our_observations(request)
        return self.deployments.handler(request)

    def _our_observations(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        by_variable: dict = {}
        for variable in payload["variable"]["dcids"]:
            for entity in payload["entity"]["dcids"]:
                provenance = self.our_facets.get((variable, entity))
                if provenance is not None:
                    block = {"orderedFacets": [{"facetId": provenance, "obsCount": 1}]}
                    by_variable.setdefault(variable, {"byEntity": {}})["byEntity"][entity] = block
        facets = {p: {"provenanceId": p} for p in set(self.our_facets.values())}
        return httpx.Response(200, json={"byVariable": by_variable, "facets": facets})

    def un_node_reads(self, since: int = 0) -> list[str]:
        """Every dcid a node request to the UN deployment named, from request `since` on."""
        return [
            dcid
            for request in self.requests[since:]
            if request.url.host == _UN_HOST and request.url.path.endswith("/v2/node")
            for dcid in json.loads(request.content)["nodes"]
        ]


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> _Wire:
    """Answers every request a client sends to either deployment. A request to any other
    host, the Datasets app under its test client among them, is sent as it would be."""
    double = _Wire()
    send = httpx.Client.send

    def routed(self: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        if request.url.host not in {_OURS_HOST, _UN_HOST}:
            return send(self, request, **kwargs)
        response = double.handler(request)
        response.request = request
        return response

    monkeypatch.setattr(httpx.Client, "send", routed)
    return double


def _live() -> LiveGraphClient:
    return LiveGraphClient(base=OURS_API.removesuffix("/core/api"), un_base_url=UN_API)


@pytest.fixture
def live(wire: _Wire) -> Iterator[LiveGraphClient]:
    with _live() as client:
        yield client


@pytest.fixture
def origin(wire: _Wire) -> Iterator[OriginGraphClient]:
    with build_graph_client(
        base_url=OURS_API,
        token=None,
        un_base_url=UN_API,
        un_availability=SdmxClient(base_url=UN_API),
        max_attempts=1,
    ) as client:
        yield client


def _search(graph: LiveGraphClient) -> Spec:
    """The Spec Search answers the query with at Kenya, from our pass alone."""
    llm = _ElectFirstLLM()
    result = asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(_PLACES),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=llm,
            base_timing={},
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(deadline=None),
        )
    )
    assert result.status == "definite", result.no_data_reason
    assert len(llm.prompts) == 1
    (spec,) = result.specs
    return spec


def _served(spec: Spec) -> list[str]:
    return [member.ref.ref for member in spec.resolved.members]


def _resubmit(graph: LiveGraphClient, spec: Spec) -> RegionResult:
    return resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(identity=spec.definition.identity),
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=EngineOptions(materialise_graph=graph),
    )


def _datasets(graph: OriginGraphClient, catalogue: CatalogueSnapshot) -> TestClient:
    store = FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=_sdmx_never_called(),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage(),
        catalogue=fake_catalogue_service(catalogue),
    )


def _mint(client: TestClient, sidecar: dict) -> httpx.Response:
    body = {
        "schema_version": 1,
        "input": {"kind": "spec_id", "spec_id": sidecar["spec_id"], "definition": sidecar},
        "options": {"bundle": False},
    }
    return client.post("/api/datasets", json=body)


# -- a first answer and its resubmit --


def test_a_resubmitted_un_identity_returns_the_first_answers_status_and_identity(
    live: LiveGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_who_catalogue())
    spec = _search(live)
    assert _served(spec) == [UN_VARIABLE]

    again = _resubmit(live, spec)

    assert again.status == "definite", again.no_data_reason
    (resubmitted,) = again.specs
    assert (resubmitted.spec_id, resubmitted.definition) == (spec.spec_id, spec.definition)
    assert _served(resubmitted) == _served(spec)


def test_a_resubmit_reads_no_node_of_its_shape_from_the_un_deployment(
    wire: _Wire, live: LiveGraphClient, bind_catalogue
) -> None:
    """The resubmit runs on a client and an enumerator that have read nothing, so every node
    it needs is one it asks for. The facet's unit is no arc of a variable, so the UN tables
    hold no name for it and its label is the one node read the UN deployment gets."""
    bind_catalogue(_who_catalogue())
    spec = _search(live)
    shared_enumerator().clear()
    sent = len(wire.requests)

    with _live() as cold:
        again = _resubmit(cold, spec)

    assert again.status == "definite", again.no_data_reason
    assert set(wire.un_node_reads(sent)) <= {UN_UNIT}
    # The control: the resubmit did reach the UN deployment, for its observations.
    assert {request.url.path for request in wire.requests[sent:] if request.url.host == _UN_HOST}


# -- Search and the Datasets mint --


def test_every_stat_var_a_transferred_un_answer_serves_can_be_minted(
    live: LiveGraphClient, origin: OriginGraphClient, bind_catalogue
) -> None:
    catalogue = _who_catalogue()
    bind_catalogue(catalogue)
    spec = _search(live)
    served = _served(spec)
    assert served
    datasets = _datasets(origin, catalogue)

    answer = _mint(
        datasets,
        {
            "schema_version": 1,
            "spec_id": spec.spec_id,
            "identity": spec.definition.identity.model_dump(mode="json", exclude_none=True),
        },
    )
    pinned = {stat_var: _mint(datasets, _sidecar(origin, stat_var, (_KEN,))) for stat_var in served}

    assert answer.status_code == 200, answer.text
    minted = {entry["ref"] for entry in answer.json()["trace"]["resolved_stat_vars"]}
    assert {f"dcid:{stat_var}" for stat_var in served} <= minted
    assert {stat_var: response.status_code for stat_var, response in pinned.items()} == (
        dict.fromkeys(served, 200)
    )


# -- the members of a successor shape --


def test_a_successor_shape_holds_the_successors_of_the_recalled_variants(
    wire: _Wire, live: LiveGraphClient, bind_catalogue
) -> None:
    """Our resolver recalls two superseded age variants. The series holds a third age the
    recall did not find: the shape's members leave it out and its taxonomy offers it, the
    recalled ages first, with no node read on the UN deployment."""
    series, stem = "undata/who/SERIES-WIDE", "undata/who/WIDE"
    ages = ("Y0T4", "Y5T14", "Y15T24")
    successors = {f"sdg/WIDE.AGE--{age}": f"{stem}.AGE--{age}" for age in ages[:2]}
    bind_catalogue(
        restrictive_catalogue(
            provenances=[provenance_row(_WHO_ROW, UN_SYSTEM, segment="who")],
            withheld_variables=list(successors),
            successors=successors,
            un_mirror=un_mirror(
                {
                    f"{stem}.AGE--{age}": un_variable(
                        series,
                        name=f"Wide [Age={age}]",
                        constraints={"undataAge": ("Age", f"undata/AGE-{age}", age)},
                    )
                    for age in ages
                }
            ),
        )
    )
    wire.our_recall = dict(zip(successors, (0.9, 0.8), strict=True))

    recalled = live.detect_svs("wide").candidates
    (shape,) = derive_shapes(
        confirmed_svs=[c.dcid for c in recalled],
        graph=live,
        sv_scores={c.dcid: c.score for c in recalled},
    )

    assert [c.dcid for c in recalled] == list(successors.values())
    assert set(shape.sv_arc_facts or {}) == set(successors.values())
    assert (shape.slot_taxonomy or {}).get("how:undataAge") == [
        "undata/AGE-Y0T4",
        "undata/AGE-Y5T14",
        "undata/AGE-Y15T24",
    ]
    assert wire.un_node_reads() == []


# -- a superseded provenance's data on the observation read --


@pytest.mark.parametrize("un_system_enabled", [True, False], ids=["un-on", "un-off"])
def test_a_facet_under_the_superseded_provenance_is_not_served_with_the_un_origin_on_or_off(
    wire: _Wire, live: LiveGraphClient, bind_catalogue, un_system_enabled: bool
) -> None:
    wire.our_facets = {(_BASE_SDG, _KEN): _BASE_SDG_PROVENANCE, (_OURS, _KEN): _WDI}
    bind_catalogue(_refusing_catalogue(un_system_enabled=un_system_enabled))

    served = live.observation_facets_batch([_BASE_SDG, _OURS], [_KEN])

    assert served[_BASE_SDG][_KEN] == []
    # The control: the same read serves the facet an admitted base provenance holds.
    assert [facet.provenance_id for facet in served[_OURS][_KEN]] == [_WDI]


# -- the pipeline, with both refused kinds recalled --

_EXCLUDED = "undata/sdg/DC_ODA_BDVDL"
_EXCLUDED_SERIES = "undata/sdg/SERIES-DC_ODA_BDVDL"
_BASE_TOPIC = "dc/topic/sdg_3.2.1"
_SUPERSEDED_TOPIC = "dc/topic/sdg_3.2.2"
_UN_TOPIC = "undata/topic/sdgf/17.2.1"

_NODES = {
    _KEN: {"label": "Kenya", "type": "Country"},
    _OURS: _sv_node(_OURS, population_type="Person", meas_prop="literacyRate"),
    _BASE_SDG: _sv_node(_BASE_SDG, population_type="SDG_SH_DYN_MORT", meas_prop="value"),
    _SDG: _sv_node(_SDG, population_type=_SDG_SERIES, meas_prop="value"),
    _EXCLUDED: _sv_node(_EXCLUDED, population_type=_EXCLUDED_SERIES, meas_prop="value"),
    _BASE_TOPIC: _topic_node(_BASE_SDG, _OURS),
    _SUPERSEDED_TOPIC: _topic_node(_BASE_SDG),
    _UN_TOPIC: _topic_node(_EXCLUDED, _SDG),
}
_REFUSED = (_BASE_SDG, _EXCLUDED)


def _refusing_catalogue(*, un_system_enabled: bool = True) -> CatalogueSnapshot:
    """`_BASE_SDG` holds data only under `dc/base/UN_SDG` and has no successor, and
    `_EXCLUDED` is a variable of a series the SDG agency excludes. Each sits
    in a topic beside a variable the catalogue admits, and `_BASE_SDG` alone in another."""
    return restrictive_catalogue(
        provenances=[
            provenance_row(_WDI, BASE),
            provenance_row(_SDG_ROW, UN_SYSTEM, segment="sdg"),
            provenance_row(_WHO_ROW, UN_SYSTEM, segment="who"),
        ],
        base_variables={_OURS: ([_WDI], [])},
        withheld_variables=[_BASE_SDG],
        topics={_BASE_TOPIC: (_BASE_SDG, _OURS), _SUPERSEDED_TOPIC: (_BASE_SDG,)},
        un_mirror=un_mirror(
            {_SDG: un_variable(_SDG_SERIES, name=_SDG)},
            excluded={_EXCLUDED: _EXCLUDED_SERIES},
            topics={_UN_TOPIC: (_SDG,)},
        ),
        un_system_enabled=un_system_enabled,
    )


def _recalling(
    ours: Mapping[str, float], un: Mapping[str, float], *, data: tuple[str, ...]
) -> FakeGraph:
    """Each resolve returns its scored dcids, and every variable in `data` holds
    observations at Kenya."""

    def leg(scored: Mapping[str, float]) -> dict:
        return {
            "svs": list(scored),
            "cosine_scores": list(scored.values()),
            "types": {dcid: _NODES[dcid]["type"] for dcid in scored},
        }

    return FakeGraph(
        nodes=_NODES,
        obs={f"{dcid}|{_KEN}": [_FACET] for dcid in data},
        detect={_QUERY: leg(ours)},
        un_detect={_QUERY: leg(un)},
        resolve={"Kenya": _KEN},
    )


@pytest.mark.parametrize("un_system_enabled", [True, False], ids=["un-on", "un-off"])
def test_a_recalled_superseded_variable_and_an_excluded_one_reach_no_part_of_the_answer(
    bind_catalogue, un_system_enabled: bool
) -> None:
    """Both refused variables outscore every admitted one on their own resolve, are recalled
    again through a topic, and hold data at Kenya. Our admitted variable holds none, so the
    UN pass runs when the UN origin is on."""
    bind_catalogue(_refusing_catalogue(un_system_enabled=un_system_enabled))
    llm = _ElectFirstLLM()
    graph = _recalling(
        {_BASE_SDG: 0.95, _BASE_TOPIC: 0.93, _SUPERSEDED_TOPIC: 0.92, _OURS: 0.9},
        {_EXCLUDED: 0.95, _UN_TOPIC: 0.93, _SDG: 0.85},
        data=(_SDG, *_REFUSED),
    )

    result = _resolve(graph, llm=llm)

    # The controls: each pass's admitted variable is on its menu, and the UN's is elected.
    assert _OURS in llm.prompts[0]
    if un_system_enabled:
        assert len(llm.prompts) == 2 and _SDG in llm.prompts[1]
        assert _elected(result) == [_SDG]
    else:
        assert len(llm.prompts) == 1
        assert result.status == "no_data"
    answer = repr(result)
    assert [
        dcid
        for dcid in _REFUSED
        if dcid in answer or any(_names(menu, dcid) for menu in llm.prompts)
    ] == []


@pytest.mark.parametrize("un_system_enabled", [True, False], ids=["un-on", "un-off"])
def test_a_question_only_refused_variables_answer_serves_none_of_them(
    bind_catalogue, caplog, un_system_enabled: bool
) -> None:
    """Our resolve recalls only refused dcids, and the UN's a refused one and `_SDG`, which
    holds no data at Kenya. The controls: our pass ends without a shape, and with the
    UN origin on the UN pass runs and its menu holds `_SDG`."""
    bind_catalogue(_refusing_catalogue(un_system_enabled=un_system_enabled))
    llm = _ElectFirstLLM()
    graph = _recalling(
        {_BASE_SDG: 0.95, _SUPERSEDED_TOPIC: 0.9}, {_EXCLUDED: 0.95, _SDG: 0.8}, data=_REFUSED
    )

    with caplog.at_level(logging.INFO, logger="search.engine.region.pipeline"):
        result = _resolve(graph, llm=llm)

    assert (result.status, result.specs) == ("no_data", ())
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("gap_fill ")]
    assert "status=no_data/variable_not_resolved" in line
    if un_system_enabled:
        assert "un_pass=ran:" in line
        (menu,) = llm.prompts
        assert _SDG in menu
    else:
        assert llm.prompts == []
    assert [
        dcid
        for dcid in _REFUSED
        if dcid in repr(result) or any(_names(menu, dcid) for menu in llm.prompts)
    ] == []


@pytest.mark.parametrize("un_system_enabled", [True, False], ids=["un-on", "un-off"])
@pytest.mark.parametrize("dcid", [*_REFUSED, _SUPERSEDED_TOPIC])
def test_no_origin_admits_or_replaces_a_superseded_or_an_excluded_variable(
    dcid: str, un_system_enabled: bool
) -> None:
    catalogue = _refusing_catalogue(un_system_enabled=un_system_enabled)

    assert catalogue.admit_candidate(dcid) is None
    assert catalogue.successor(dcid) is None
    assert catalogue.data_commons(dcid) == ()
    # The control: the topic holding an admitted variable beside the superseded one.
    assert catalogue.admit_candidate(_BASE_TOPIC) == BASE
