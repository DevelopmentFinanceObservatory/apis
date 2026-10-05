"""`OriginGraphClient`: routing by dcid, admission, the domain filter, the
withheld read, table-backed presence, and the catalogue it reads per call."""

from __future__ import annotations

import contextvars
import json
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date

import httpx
import pytest

from dc_graph import (
    GraphCapExceededError,
    GraphClient,
    GraphDeadlineExceededError,
    GraphReadError,
)
from graph_generation import FakeGeneration
from origin_catalogue import (
    BASE,
    DFO,
    UN_SYSTEM,
    Catalogue,
    CatalogueNotBound,
    CatalogueSnapshot,
    ProvenanceRow,
    SourceRef,
    bound_catalogue,
    recording_un_blocked_places,
)
from origin_catalogue.graph import (
    OriginGraphClient,
    UnGeneration,
    build_graph_client,
    is_un_dcid,
)
from tests.origin_catalogue.fake_catalogue import restrictive_catalogue

_OURS_HOST = "graph.example.org"
_UN_HOST = "un.example.org"

_CRS = "provenance/OECDCreditorReportingSystem"
_WDI = "dc/base/WorldDevelopmentIndicators"
_CENSUS = "dc/base/USCensusPEP"
_UN_WHO = "undata/p/WHO"
_UN_UNKNOWN = "undata/p/NOT_REGISTERED"

_POPULATION = "Count_Person"
_GDP = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_CRS_VARIABLE = "Amount_FinancialFlow_Gross_Disbursement_Nominal_ODA"
_UN_MORTALITY = "undata/who/MORT_U5_RATE"
_UN_AGE = "undataAge"

_URLS = {
    _CRS: "https://data-explorer.oecd.org/crs",
    _WDI: "https://datatopics.worldbank.org/wdi",
    _CENSUS: "https://www.census.gov/pep",
    _UN_WHO: "https://data.who.int/gho",
    _UN_UNKNOWN: "https://example.org/unknown",
}


def _row(provenance_id: str, origin, *, segment: str | None = None) -> ProvenanceRow:
    return ProvenanceRow(
        id=provenance_id,
        data_commons=origin,
        source=SourceRef(ref=provenance_id, label=provenance_id),
        name=provenance_id,
        url=_URLS[provenance_id],
        segment=segment,
    )


def _catalogue(
    *,
    un_system_enabled: bool = True,
    build_id: str = "build-a",
    un_blocked_places: tuple[str, ...] = (),
    overlaps: tuple[str, ...] = (),
) -> CatalogueSnapshot:
    """Admits CRS (DFO), WDI (base) and WHO (UN). The US Census provenance is not admitted.
    `_GDP` holds data at KEN; KEN and USA are scan places, `geoId/06` is not. `overlaps` are
    base variables DFO also holds."""
    return restrictive_catalogue(
        provenances=[_row(_CRS, DFO), _row(_WDI, BASE), _row(_UN_WHO, UN_SYSTEM, segment="who")],
        dfo_variables=[_CRS_VARIABLE, *overlaps],
        base_variables={_GDP: ([_WDI], ["country/KEN"]), _POPULATION: ([_WDI], ["country/USA"])},
        scan_places=["country/KEN", "country/USA"],
        un_blocked_places=un_blocked_places,
        un_system_enabled=un_system_enabled,
        build_id=build_id,
    )


@dataclass
class _Graph:
    """Both deployments behind one mock transport, told apart by host.

    `labels` answers node reads; `facets` maps `(variable, entity)` to provenance ids, one facet
    each. Our mixer honours `filter.domains` as the real one does, by hostname suffix. A GET
    naming a place in `refused` gets the UN front door's 403 refusal page, as the real one does
    for some places.
    """

    labels: dict[str, str] = field(default_factory=dict)
    facets: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)
    threads: list[tuple[str, str]] = field(default_factory=list)
    refused: set[str] = field(default_factory=set)
    un_observation_status: int = 200
    on_request: Callable[[httpx.Request], None] | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.threads.append((request.url.host, threading.current_thread().name))
        if self.on_request is not None:
            self.on_request(request)
        if request.url.path.endswith("/v2/resolve"):
            return httpx.Response(200, json={"entities": []})
        if (
            request.url.host == _UN_HOST
            and request.url.path.endswith("/v2/observation")
            and self.un_observation_status != 200
        ):
            return httpx.Response(self.un_observation_status, json={"error": "unavailable"})
        if request.method == "GET" and self.refused & set(
            request.url.params.get_list("entity.dcids")
        ):
            return httpx.Response(
                403,
                text="<!doctype html><title>403</title>403 Forbidden",
                headers={"content-type": "text/html; charset=UTF-8"},
            )
        if request.url.path.endswith("/v2/node"):
            payload = json.loads(request.content)
            return httpx.Response(200, json=self._node(payload["nodes"], payload["property"]))
        return httpx.Response(200, json=self._observation(request))

    def _node(self, nodes: list[str], prop: str) -> dict:
        data: dict = {}
        for dcid in nodes:
            label = self.labels.get(dcid)
            if label is None:
                data[dcid] = {}
                continue
            name = {"nodes": [{"value": label}]}
            if prop == "->*":
                data[dcid] = {"arcs": {"name": name, "rangeIncludes": {"nodes": [{"dcid": "C"}]}}}
            elif prop == "<-typeOf":
                data[dcid] = {"arcs": {"typeOf": {"nodes": [{"dcid": f"{dcid}/member"}]}}}
            else:
                data[dcid] = {"arcs": {"name": name}}
        return {"data": data}

    def _observation(self, request: httpx.Request) -> dict:
        if request.method == "POST":
            payload = json.loads(request.content)
            variables = payload["variable"]["dcids"]
            entities = payload["entity"]["dcids"]
            select = payload["select"]
            domains = payload.get("filter", {}).get("domains")
        else:
            params = request.url.params
            variables = params.get_list("variable.dcids")
            entities = params.get_list("entity.dcids")
            select = params.get_list("select")
            domains = params.get_list("filter.domains") or None
        by_variable: dict = {}
        facet_map: dict = {}
        for variable in variables:
            by_entity: dict = {}
            for entity in entities:
                held = [
                    pid
                    for pid in self.facets.get((variable, entity), [])
                    if "facet" not in select
                    or domains is None
                    or any(httpx.URL(_URLS[pid]).host.endswith(domain) for domain in domains)
                ]
                if not held:
                    continue
                if "facet" not in select:
                    by_entity[entity] = {}
                    continue
                blocks = []
                for pid in held:
                    facet_map[pid] = {"provenanceId": pid}
                    blocks.append({"facetId": pid, "obsCount": 1})
                by_entity[entity] = {"orderedFacets": blocks}
            if by_entity:
                by_variable[variable] = {"byEntity": by_entity}
        return {"byVariable": by_variable, "facets": facet_map}

    def on(self, host: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.host == host]

    def observation_requests(self, host: str) -> list[httpx.Request]:
        return [r for r in self.on(host) if r.url.path.endswith("/v2/observation")]

    def dcids_sent(self, host: str) -> list[str]:
        sent: list[str] = []
        for request in self.on(host):
            if request.method == "POST":
                payload = json.loads(request.content)
                sent.extend(payload.get("nodes", []))
                sent.extend(payload.get("variable", {}).get("dcids", []))
            else:
                sent.extend(request.url.params.get_list("variable.dcids"))
        return sent


def _client(graph: _Graph, *, today: Callable[[], date] | None = None) -> OriginGraphClient:
    transport = httpx.MockTransport(graph.handler)
    return OriginGraphClient(
        ours=GraphClient(
            base_url=f"https://{_OURS_HOST}/core/api",
            token="secret",
            transport=transport,
            generation=FakeGeneration(token="rev-1"),
            obs_cache_size=64,
        ),
        un=GraphClient(
            base_url=f"https://{_UN_HOST}/core/api",
            transport=transport,
            generation=UnGeneration(today=today),
            observation_get_url_budget_bytes=7_000,
            obs_cache_size=64,
        ),
    )


@pytest.fixture
def graph() -> _Graph:
    return _Graph(
        labels={
            "country/KEN": "Kenya",
            "country/USA": "United States",
            _UN_MORTALITY: "Under-five mortality",
            _UN_AGE: "Age",
            "C": "Age group",
        },
        facets={
            (_POPULATION, "country/KEN"): [_CRS, _WDI, _CENSUS],
            (_POPULATION, "country/USA"): [_WDI, _CENSUS],
            (_POPULATION, "geoId/06"): [_CENSUS],
            (_GDP, "geoId/06"): [_WDI],
            (_GDP, "geoId/08"): [_CENSUS],
            (_CRS_VARIABLE, "country/KEN"): [_CRS],
            (_UN_MORTALITY, "country/KEN"): [_UN_WHO, _UN_UNKNOWN],
            (_UN_MORTALITY, "country/USA"): [_UN_UNKNOWN],
        },
    )


@pytest.fixture
def client(graph: _Graph) -> Iterator[OriginGraphClient]:
    with _client(graph) as origin:
        yield origin


def _providers(facets: dict[str, list]) -> dict[str, list[str | None]]:
    return {entity: [f.provenance_id for f in listed] for entity, listed in facets.items()}


# -- routing --


@pytest.mark.parametrize(
    ("dcid", "expected"),
    [
        (_UN_MORTALITY, True),
        ("undata/p/WHO", True),
        (_UN_AGE, True),
        ("undataCoo", True),
        ("undatabase", False),
        (_POPULATION, False),
        ("dc/base/UN_SDG", False),
        ("country/KEN", False),
    ],
)
def test_un_dcids_are_undata_nodes_and_undata_properties(dcid: str, expected: bool) -> None:
    assert is_un_dcid(dcid) is expected


def test_batch_reads_split_by_origin_and_merge_in_first_occurrence_order(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())

    labels = client.node_labels_batch(
        [_UN_MORTALITY, "country/KEN", _UN_AGE, "country/USA", _UN_MORTALITY]
    )

    assert list(labels) == [_UN_MORTALITY, "country/KEN", _UN_AGE, "country/USA"]
    assert graph.dcids_sent(_OURS_HOST) == ["country/KEN", "country/USA"]
    assert graph.dcids_sent(_UN_HOST) == [_UN_MORTALITY, _UN_AGE]


def test_node_arcs_many_and_names_keep_every_requested_dcid_in_order(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())

    names = client.node_names_many(["country/USA", _UN_AGE, "country/KEN"])
    arcs = client.node_arcs_many([_UN_AGE, "country/KEN"], properties=["name"])

    assert names == {
        "country/USA": ("United States",),
        _UN_AGE: ("Age",),
        "country/KEN": ("Kenya",),
    }
    assert list(names) == ["country/USA", _UN_AGE, "country/KEN"]
    assert arcs == {_UN_AGE: {"name": ("Age",)}, "country/KEN": {"name": ("Kenya",)}}


def test_enum_members_route_on_the_property(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())

    assert client.enum_members(_UN_AGE) == ["C/member"]
    assert graph.on(_OURS_HOST) == []


def test_reads_that_never_carry_un_dcids_go_to_our_mixer(
    graph: _Graph, client: OriginGraphClient
) -> None:
    client.post("/v2/node", {"nodes": ["country/KEN"], "property": "->name"})
    client.resolve_by_arc(["Kenya"], arc="<-description->dcid")

    assert len(graph.on(_OURS_HOST)) == 2
    assert graph.on(_UN_HOST) == []


_UN_DEBT_SERIES = "undata/sdg/SERIES-DT_DOD_DECT_GN_ZS"


@pytest.mark.parametrize(
    "filters",
    [
        pytest.param(
            {"populationType": _UN_DEBT_SERIES, "statType": "measuredValue"}, id="un_value_first"
        ),
        pytest.param(
            {"statType": "measuredValue", "populationType": _UN_DEBT_SERIES}, id="un_value_last"
        ),
        pytest.param({"statType": "measuredValue", _UN_AGE: "Years15To24"}, id="un_filter_key"),
    ],
)
def test_a_reverse_read_filtered_on_a_un_name_goes_to_the_un(
    graph: _Graph, client: OriginGraphClient, filters: dict[str, str]
) -> None:
    client.node_subjects("value", property="measuredProperty", filters=filters, max_results=10)

    assert len(graph.on(_UN_HOST)) == 1
    assert graph.on(_OURS_HOST) == []


def test_a_reverse_read_naming_no_un_node_goes_to_our_mixer(
    graph: _Graph, client: OriginGraphClient
) -> None:
    client.node_subjects(
        "count",
        property="measuredProperty",
        filters={"populationType": "Person", "statType": "measuredValue"},
        max_results=10,
    )
    client.node_subjects("count", property="measuredProperty", max_results=10)

    assert len(graph.on(_OURS_HOST)) == 2
    assert graph.on(_UN_HOST) == []


def test_un_disabled_answers_a_un_filtered_reverse_read_empty_and_sends_nothing(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue(un_system_enabled=False))

    page = client.node_subjects(
        "value",
        property="measuredProperty",
        filters={"populationType": _UN_DEBT_SERIES, "statType": "measuredValue"},
        max_results=10,
    )

    assert (page.dcids, page.truncated) == ((), False)
    assert graph.requests == []


def test_un_disabled_answers_un_dcids_empty_and_sends_them_nowhere(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue(un_system_enabled=False))

    assert client.node_label(_UN_MORTALITY) is None
    assert client.node_arcs(_UN_AGE) is None
    assert client.enum_members(_UN_AGE) == []
    assert client.node_labels_batch([_UN_MORTALITY, "country/KEN"]) == {"country/KEN": "Kenya"}
    assert client.node_names_many([_UN_AGE, "country/KEN"]) == {
        _UN_AGE: (),
        "country/KEN": ("Kenya",),
    }
    assert client.node_arcs_batch([_UN_AGE])[_UN_AGE] is None
    assert client.node_subjects_many(["country/KEN"], property=_UN_AGE, max_results=5) == {
        "country/KEN": ((), "complete")
    }
    facets = client.observation_facets_batch([_UN_MORTALITY], ["country/KEN"])
    assert facets == {_UN_MORTALITY: {"country/KEN": []}}
    assert client.entities_with_data([_UN_MORTALITY], ["country/KEN"]) == []
    assert client.withheld_provenances([_UN_MORTALITY], ["country/KEN"]) == ()

    assert graph.on(_UN_HOST) == []
    assert not any(is_un_dcid(dcid) for dcid in graph.dcids_sent(_OURS_HOST))


# -- admission and the domain filter --


def test_facets_are_admitted_with_dfo_precedence_per_place(
    client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())

    facets = client.observation_facets_batch(
        [_POPULATION, _UN_MORTALITY], ["country/KEN", "country/USA", "geoId/06"]
    )

    assert _providers(facets[_POPULATION]) == {
        "country/KEN": [_CRS],
        "country/USA": [_WDI],
        "geoId/06": [],
    }
    assert _providers(facets[_UN_MORTALITY]) == {
        "country/KEN": [_UN_WHO],
        "country/USA": [],
        "geoId/06": [],
    }


def test_our_facet_reads_send_the_admitted_hostnames_and_the_un_reads_none(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())

    client.observation_facets_batch_many(
        stat_vars=[_POPULATION, _UN_MORTALITY], entities=["country/KEN"]
    )

    [ours] = graph.observation_requests(_OURS_HOST)
    assert json.loads(ours.content)["filter"] == {
        "domains": ["data-explorer.oecd.org", "datatopics.worldbank.org"]
    }
    [un] = graph.observation_requests(_UN_HOST)
    assert un.method == "GET"
    assert "filter.domains" not in un.url.params


def test_the_domain_filter_is_part_of_the_observation_cache_key(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())
    client.observation_facets_batch([_POPULATION], ["country/USA"])
    client.observation_facets_batch([_POPULATION], ["country/USA"])
    assert len(graph.observation_requests(_OURS_HOST)) == 1

    wider = restrictive_catalogue(
        provenances=[_row(_WDI, BASE), _row(_CENSUS, BASE)], scan_places=[]
    )
    bind_catalogue(wider)
    facets = client.observation_facets_batch([_POPULATION], ["country/USA"])

    assert len(graph.observation_requests(_OURS_HOST)) == 2
    assert _providers(facets[_POPULATION]) == {"country/USA": [_WDI, _CENSUS]}


def test_withheld_provenances_reads_once_without_a_filter(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())

    withheld = client.withheld_provenances(
        [_POPULATION], ["country/KEN", "country/USA", "geoId/06"]
    )

    assert withheld == (_CENSUS,)
    [request] = graph.observation_requests(_OURS_HOST)
    assert "filter" not in json.loads(request.content)
    assert client.withheld_provenances([_UN_MORTALITY], ["country/USA"]) == (_UN_UNKNOWN,)


def test_withheld_provenances_reads_each_backend_once_for_many_variables(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())

    withheld = client.withheld_provenances(
        [_UN_MORTALITY, _POPULATION, _UN_MORTALITY], ["country/KEN", "country/USA"]
    )

    assert withheld == (_UN_UNKNOWN,)
    assert len(graph.observation_requests(_OURS_HOST)) == 1
    assert len(graph.observation_requests(_UN_HOST)) == 1


def test_withheld_at_names_only_pairs_with_no_admitted_facet(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    """USA holds admitted WDI beside Census, so nothing is withheld there; `geoId/06` holds
    only Census for population and only admitted WDI for GDP; `geoId/08` holds only Census."""
    bind_catalogue(_catalogue())
    places = ["country/USA", "geoId/06", "geoId/08", "country/KEN"]

    withheld = client.withheld_at([_POPULATION, _GDP, _UN_MORTALITY], places)

    assert withheld == {
        "country/USA": (_UN_UNKNOWN,),
        "geoId/06": (_CENSUS,),
        "geoId/08": (_CENSUS,),
    }
    assert client.withheld_provenances([_POPULATION, _GDP, _UN_MORTALITY], places) == (
        _UN_UNKNOWN,
        _CENSUS,
    )


def test_withheld_reads_carry_the_deadline_and_byte_cap(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())

    with pytest.raises(GraphCapExceededError):
        client.withheld_at([_POPULATION], ["country/KEN"], max_bytes=10)
    with pytest.raises(GraphDeadlineExceededError):
        client.withheld_provenances([_POPULATION], ["country/KEN"], deadline=time.monotonic() - 1.0)


# -- presence --


def test_base_presence_answers_scan_places_from_the_table_and_reads_the_rest(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())
    places = ["geoId/08", "country/USA", "geoId/06", "country/KEN"]

    present = client.entities_with_data([_GDP], places, max_bytes=1_000_000)

    assert present == ["geoId/06", "country/KEN"]
    [request] = graph.observation_requests(_OURS_HOST)
    payload = json.loads(request.content)
    assert payload["entity"]["dcids"] == ["geoId/08", "geoId/06"]
    assert payload["select"] == ["variable", "entity", "facet"]
    assert payload["filter"] == {"domains": ["data-explorer.oecd.org", "datatopics.worldbank.org"]}


def test_presence_reads_nothing_when_the_table_covers_every_place(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())

    assert client.entities_with_data([_GDP, _POPULATION], ["country/KEN", "country/USA"]) == [
        "country/KEN",
        "country/USA",
    ]
    assert graph.requests == []


def test_dfo_presence_reads_admitted_facets_and_skips_places_with_only_excluded_data(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    """`_POPULATION` is an overlap: USA answers from the table and KEN holds DFO's CRS facet.
    At `geoId/06` both variables hold only the non-admitted Census facet, as a DFO variable
    an excluded base provenance also holds does; a plain presence read would count it."""
    bind_catalogue(_catalogue(overlaps=(_POPULATION,)))
    graph.facets[(_CRS_VARIABLE, "geoId/06")] = [_CENSUS]
    places = ["geoId/06", "country/USA", "country/KEN"]

    assert client.entities_with_data([_POPULATION], places) == ["country/USA", "country/KEN"]
    assert client.entities_with_data([_CRS_VARIABLE], places) == ["country/KEN"]
    requests = [json.loads(r.content) for r in graph.observation_requests(_OURS_HOST)]
    assert [r["entity"]["dcids"] for r in requests] == [
        ["geoId/06", "country/KEN"],
        places,
    ]
    assert all(r["select"][-1] == "facet" and "filter" in r for r in requests)


def test_presence_by_variable_admits_each_variable_at_its_own_places(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    """`_GDP` answers Kenya from the table and holds admitted WDI at `geoId/06` and only Census
    at `geoId/08`. The UN variable holds an admitted WHO facet at Kenya and only an unregistered
    provenance at the USA."""
    bind_catalogue(_catalogue())
    places = ["geoId/06", "geoId/08", "country/USA", "country/KEN"]

    present = client.entities_with_data_by_variable([_GDP, _CRS_VARIABLE, _UN_MORTALITY], places)

    assert present == {
        _GDP: ["geoId/06", "country/KEN"],
        _CRS_VARIABLE: ["country/KEN"],
        _UN_MORTALITY: ["country/KEN"],
    }
    assert _UN_MORTALITY not in graph.dcids_sent(_OURS_HOST)
    assert set(graph.dcids_sent(_UN_HOST)) == {_UN_MORTALITY}


def test_presence_by_variable_with_no_variable_or_no_entity_reads_nothing(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())

    assert client.entities_with_data_by_variable([], ["country/KEN"]) == {}
    assert client.entities_with_data_by_variable([_CRS_VARIABLE], []) == {}
    assert graph.requests == []


@pytest.mark.parametrize("variable", [_CRS_VARIABLE, _UN_MORTALITY], ids=["ours", "un"])
def test_presence_by_variable_reads_carry_the_deadline(
    client: OriginGraphClient, bind_catalogue, variable: str
) -> None:
    bind_catalogue(_catalogue())

    with pytest.raises(GraphDeadlineExceededError):
        client.entities_with_data_by_variable(
            [variable], ["geoId/06"], deadline=time.monotonic() - 1.0
        )


# -- a UN failure inside a read naming our variables --


def test_a_mixed_facet_read_serves_ours_and_records_the_un_failure(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())
    graph.un_observation_status = 404
    places = ["country/KEN", "country/USA"]

    with recording_un_blocked_places() as record:
        facets = client.observation_facets_batch([_POPULATION, _UN_MORTALITY], places)

    assert _providers(facets[_POPULATION]) == {"country/KEN": [_CRS], "country/USA": [_WDI]}
    assert facets[_UN_MORTALITY] == {"country/KEN": [], "country/USA": []}
    [failure] = record.failures()
    assert (failure.variables, failure.places) == ((_UN_MORTALITY,), tuple(places))
    assert "404" in failure.detail
    assert record.places() == ()


def test_mixed_presence_and_withheld_reads_serve_ours_and_record_the_un_failure(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())
    graph.un_observation_status = 404

    with recording_un_blocked_places() as record:
        present = client.entities_with_data([_GDP, _UN_MORTALITY], ["geoId/06", "country/USA"])
        withheld = client.withheld_provenances([_POPULATION, _UN_MORTALITY], ["geoId/06"])

    assert present == ["geoId/06"]
    assert withheld == (_CENSUS,)
    assert [(f.variables, f.places) for f in record.failures()] == [
        ((_UN_MORTALITY,), ("country/USA",)),
        ((_UN_MORTALITY,), ("geoId/06",)),
    ]


def test_a_un_failure_raises_without_a_record_or_when_the_read_names_only_un_variables(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())
    graph.un_observation_status = 404

    with pytest.raises(GraphReadError):
        client.observation_facets_batch([_POPULATION, _UN_MORTALITY], ["country/KEN"])
    with recording_un_blocked_places() as record, pytest.raises(GraphReadError):
        client.observation_facets_batch([_UN_MORTALITY], ["country/KEN"])
    assert record.failures() == ()


# -- the catalogue each call reads --


def test_a_call_admits_with_the_catalogue_bound_at_entry(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())
    admits_the_unknown_agency = restrictive_catalogue(
        provenances=[_row(_UN_UNKNOWN, UN_SYSTEM, segment="unknown")], scan_places=[]
    )
    graph.on_request = lambda _request: bind_catalogue(admits_the_unknown_agency)

    facets = client.observation_facets_batch([_UN_MORTALITY], ["country/USA"])

    assert _providers(facets[_UN_MORTALITY]) == {"country/USA": []}


def test_both_halves_run_at_once_and_admit_with_the_callers_catalogue(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())

    facets = client.observation_facets_batch_many(
        stat_vars=[_POPULATION, _UN_MORTALITY], entities=["country/KEN"]
    )

    assert _providers(facets[_UN_MORTALITY]) == {"country/KEN": [_UN_WHO]}
    threads = dict(graph.threads)
    assert threads[_OURS_HOST] == threading.current_thread().name
    assert threads[_UN_HOST].startswith("origin-un")


def test_a_read_from_a_pool_without_the_callers_context_refuses(
    client: OriginGraphClient,
) -> None:
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.node_labels_batch, ["country/KEN"])
        with pytest.raises(CatalogueNotBound):
            future.result()


# -- caches --


def test_un_reads_cache_for_the_rest_of_the_utc_day(graph: _Graph, bind_catalogue) -> None:
    bind_catalogue(_catalogue())
    today = [date(2026, 9, 29)]
    with _client(graph, today=lambda: today[0]) as origin:
        origin.observation_facets_batch([_UN_MORTALITY], ["country/KEN"])
        origin.observation_facets_batch([_UN_MORTALITY], ["country/KEN"])
        assert len(graph.observation_requests(_UN_HOST)) == 1

        today[0] = date(2026, 9, 30)
        origin.observation_facets_batch([_UN_MORTALITY], ["country/KEN"])

    assert len(graph.observation_requests(_UN_HOST)) == 2


def test_un_generation_keys_on_the_bound_build_and_utc_date() -> None:
    generation = UnGeneration(today=lambda: date(2026, 9, 29))

    with bound_catalogue(_catalogue(build_id="build-a")):
        assert generation.stamp().key == "build-a+2026-09-29"
    # A fresh context has no catalogue bound.
    with pytest.raises(CatalogueNotBound):
        contextvars.Context().run(generation.stamp)


def test_a_request_bound_to_an_older_build_seeds_nothing_into_the_newer_builds_cache(
    graph: _Graph, bind_catalogue
) -> None:
    with _client(graph) as origin:
        # A request bound to build A finishes its read after the service has loaded build B.
        bind_catalogue(_catalogue(build_id="build-a", un_blocked_places=("country/USA",)))
        origin.observation_facets_batch([_UN_MORTALITY], ["country/KEN"])

        bind_catalogue(_catalogue(build_id="build-b"))
        assert origin.un_blocked_places() == frozenset()
        origin.observation_facets_batch([_UN_MORTALITY], ["country/KEN", "country/USA"])

    last = graph.observation_requests(_UN_HOST)[-1]
    assert set(last.url.params.get_list("entity.dcids")) == {"country/KEN", "country/USA"}


# -- the factory --


def test_factory_sends_the_token_to_our_mixer_only_and_reads_the_un_by_get(
    graph: _Graph, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())
    with build_graph_client(
        base_url=f"https://{_OURS_HOST}/core/api",
        token="secret",
        un_base_url=f"https://{_UN_HOST}/core/api",
        transport=httpx.MockTransport(graph.handler),
    ) as origin:
        origin.observation_facets_batch([_POPULATION, _UN_MORTALITY], ["country/KEN"])

    [ours] = graph.on(_OURS_HOST)
    [un] = graph.on(_UN_HOST)
    assert ours.headers["x-graph-token"] == "secret"
    assert "x-graph-token" not in un.headers
    assert un.method == "GET"


def test_the_catalogue_protocol_carries_what_routing_reads() -> None:
    catalogue: Catalogue = _catalogue(un_system_enabled=False)

    assert catalogue.un_system_enabled is False
    assert catalogue.mixer_domains == ("data-explorer.oecd.org", "datatopics.worldbank.org")
    assert catalogue.is_scan_place("country/KEN")
    assert not catalogue.is_scan_place("geoId/06")


def test_places_the_un_front_door_refuses_are_served_empty_and_exposed(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())
    graph.refused = {"country/USA"}

    facets = client.observation_facets_batch(
        [_POPULATION, _UN_MORTALITY], ["country/KEN", "country/USA"]
    )

    assert _providers(facets[_UN_MORTALITY]) == {"country/KEN": [_UN_WHO], "country/USA": []}
    assert _providers(facets[_POPULATION]) == {"country/KEN": [_CRS], "country/USA": [_WDI]}
    assert client.un_blocked_places() == {"country/USA"}


def test_places_the_catalogue_found_refused_are_never_sent_to_the_un(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue(un_blocked_places=("country/USA",)))
    graph.refused = {"country/USA"}

    facets = client.observation_facets_batch([_UN_MORTALITY], ["country/KEN", "country/USA"])
    present = client.entities_with_data([_UN_MORTALITY], ["country/KEN", "country/USA"])

    assert _providers(facets[_UN_MORTALITY]) == {"country/KEN": [_UN_WHO], "country/USA": []}
    assert present == ["country/KEN"]
    sent = graph.observation_requests(_UN_HOST)
    assert len(sent) == 2
    assert all(r.url.params.get_list("entity.dcids") == ["country/KEN"] for r in sent)
    assert client.un_blocked_places() == {"country/USA"}


# -- recall resolves and refused places --


def test_resolve_indicators_posts_to_the_named_deployment(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())

    client.resolve_indicators("child mortality", target="base_and_custom")
    client.resolve_indicators("child mortality", target="custom_only", un=True)

    (ours,) = graph.on(_OURS_HOST)
    (un,) = graph.on(_UN_HOST)
    assert json.loads(ours.content) == {
        "nodes": ["child mortality"],
        "resolver": "indicator",
        "target": "base_and_custom",
    }
    assert json.loads(un.content)["target"] == "custom_only"
    assert "x-graph-token" not in un.headers


def test_a_un_resolve_with_the_un_off_sends_nothing(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue(un_system_enabled=False))

    body = client.resolve_indicators("child mortality", target="custom_only", un=True)

    assert body == {"entities": []}
    assert graph.requests == []


def test_un_reads_note_refused_places_for_the_callers_record(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())
    graph.refused = {"country/USA"}

    with recording_un_blocked_places() as record:
        client.observation_facets_batch([_POPULATION], ["country/USA"])
        assert record.places() == ()
        client.observation_facets_batch(
            [_POPULATION, _UN_MORTALITY], ["country/KEN", "country/USA"]
        )
        client.entities_with_data([_UN_MORTALITY], ["country/USA", "country/KEN"])

    assert record.places() == ("country/USA",)


def test_a_refused_place_the_catalogue_found_is_noted_once_a_read_names_it(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue(un_blocked_places=("country/USA",)))

    with recording_un_blocked_places() as record:
        client.entities_with_data([_UN_MORTALITY], ["country/KEN"])
        assert record.places() == ()
        client.entities_with_data([_UN_MORTALITY], ["country/KEN", "country/USA"])

    assert record.places() == ("country/USA",)
    client.observation_facets_batch([_UN_MORTALITY], ["country/USA"])
    assert record.places() == ("country/USA",)
