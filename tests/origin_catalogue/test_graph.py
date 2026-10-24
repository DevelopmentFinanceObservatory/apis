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
    parse_arcs,
)
from graph_generation import FakeGeneration
from graph_release.context import bound_release
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
from origin_catalogue.arc_tables import EMPTY_ARC_TABLES, ArcTables
from origin_catalogue.graph import (
    OriginGraphClient,
    UnGeneration,
    build_graph_client,
    is_un_dcid,
)
from origin_catalogue.topic_walk import walk_topics
from tests.graph_release.support import origin_release
from tests.origin_catalogue.fake_catalogue import restrictive_catalogue, un_mirror, un_variable
from tests.origin_catalogue.support import build_arc_tables

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
_UN_SERIES = "undata/who/SERIES-MORT_U5_RATE"
_UN_MORTALITY_FEMALE = "undata/who/MORT_U5_RATE.SEX--F"
_UN_SEX = "undataSex"
_UN_FEMALE = "undata/SEX-F"
_UN_TOPIC = "undata/topic/who/MORT_U5_RATE"
_UN_AID = "undata/who/ODA_1_3"
_UN_AID_SERIES = "undata/who/SERIES-ODA_1_3"
_UN_UNIT = "undata/UNIT_MEASURE-PT"

_UN_TABLES = un_mirror(
    {
        _UN_MORTALITY: un_variable(_UN_SERIES, name="Under-five mortality"),
        _UN_MORTALITY_FEMALE: un_variable(
            _UN_SERIES,
            name="Under-five mortality [Sex=Female]",
            constraints={_UN_SEX: ("Sex", _UN_FEMALE, "Female")},
        ),
    },
    excluded={_UN_AID: _UN_AID_SERIES},
    topics={_UN_TOPIC: (_UN_MORTALITY, _UN_MORTALITY_FEMALE)},
    superseded_by=_CRS,
)

_OUR_TOPIC = "dc/topic/Economy"
_UNLISTED = "Count_Person_Female"
_HOUSEHOLDS = "Count_Household"
_RECIPIENT = "financingRecipient"
_FLOW = "FinancialFlow"
_AMOUNT = "amount"
_STATVAR = {"dcid": "StatisticalVariable", "name": "StatisticalVariable", "types": ["Class"]}

_CRS_ARCS = {
    "typeOf": {"nodes": [_STATVAR, _STATVAR]},
    "name": {"nodes": [{"value": "Gross ODA disbursements"}]},
    "constraintProperties": {
        "nodes": [
            {"dcid": _RECIPIENT, "name": "Financing recipient", "types": ["Property"]},
            {"dcid": "sourceIndicatorCode", "types": ["Property"]},
        ]
    },
    "observationProperties": {"nodes": [{"dcid": _RECIPIENT, "types": ["Property"]}]},
    "populationType": {"nodes": [{"dcid": _FLOW, "name": "Financial flow"}]},
    "measuredProperty": {"nodes": [{"dcid": _AMOUNT, "types": ["Property", "Thing"]}]},
    "measurementDenominator": {"nodes": [{"dcid": _HOUSEHOLDS}]},
    "statType": {"nodes": [{"dcid": "measuredValue", "types": ["Property"]}]},
}
"""DFO's CRS variable: two `typeOf` arcs a redeclaring import repeats, an import-added
constraint property and a denominator."""

_ARCS = build_arc_tables(
    dfo={_CRS_VARIABLE: _CRS_ARCS},
    base={
        _GDP: {
            "typeOf": {"nodes": [_STATVAR]},
            "name": {"nodes": [{"value": "GDP"}, {"value": "Gross domestic product"}]},
            "alternateName": {"nodes": [{"value": "Economic output"}]},
        },
        _POPULATION: None,
    },
    schema={
        _HOUSEHOLDS: {"typeOf": {"nodes": [_STATVAR]}},
        _RECIPIENT: {
            "name": {"nodes": [{"value": "Financing recipient"}]},
            "rangeIncludes": {"nodes": [{"dcid": "Country", "types": ["Class"]}]},
        },
        "Country": {"subClassOf": {"nodes": [{"dcid": "Place", "types": ["Class"]}]}},
        "Place": {},
    },
)
"""DFO's CRS variable (`_CRS_ARCS`); GDP's names; the population variable held empty; and the
schema the CRS variable's role and denominator depend on."""

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
    arcs: ArcTables = EMPTY_ARC_TABLES,
) -> CatalogueSnapshot:
    """Admits CRS (DFO), WDI (base) and WHO (UN). The US Census provenance is not admitted.
    `_GDP` holds data at KEN; KEN and USA are scan places, `geoId/06` is not. `overlaps` are
    base variables DFO also holds. The UN tables hold `_UN_MORTALITY`, its by-sex variant
    and a topic reaching both, and exclude `_UN_AID`. The topic table holds `_OUR_TOPIC`.
    `arcs` are the arc tables, none by default."""
    return restrictive_catalogue(
        provenances=[_row(_CRS, DFO), _row(_WDI, BASE), _row(_UN_WHO, UN_SYSTEM, segment="who")],
        dfo_variables=[_CRS_VARIABLE, *overlaps],
        base_variables={_GDP: ([_WDI], ["country/KEN"]), _POPULATION: ([_WDI], ["country/USA"])},
        topics={_OUR_TOPIC: (_GDP, _POPULATION, _UNLISTED)},
        scan_places=["country/KEN", "country/USA"],
        un_blocked_places=un_blocked_places,
        un_mirror=_UN_TABLES,
        arcs=arcs,
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


def _client(
    graph: _Graph, *, today: Callable[[], date] | None = None, un_budget: int = 7_000
) -> OriginGraphClient:
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
            observation_get_url_budget_bytes=un_budget,
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
def released_crs() -> Iterator[None]:
    """Binds a release holding DFO's CRS variable with `_CRS_ARCS`, the arcs the catalogue's
    DFO table holds for it, as a release and a catalogue build of one import do."""
    with bound_release(origin_release({_CRS_VARIABLE: _CRS_ARCS})):
        yield


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
    # The UN tables name the variable, and no arc of theirs names `undataAge`.
    assert graph.dcids_sent(_UN_HOST) == [_UN_AGE]


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


# -- UN node facts from the catalogue's UN tables --


def test_a_un_variables_facts_come_from_the_un_tables_with_no_un_request(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())
    members = [_UN_MORTALITY, _UN_MORTALITY_FEMALE]

    arcs = client.node_arcs_batch(members)
    labels = client.node_labels_batch([*members, _UN_SEX, _UN_FEMALE, _UN_SERIES])

    assert parse_arcs(arcs[_UN_MORTALITY_FEMALE] or {}, ["populationType", _UN_SEX, "name"]) == {
        "populationType": (_UN_SERIES,),
        _UN_SEX: (_UN_FEMALE,),
        "name": ("Under-five mortality [Sex=Female]",),
    }
    assert client.node_arcs(_UN_MORTALITY) == arcs[_UN_MORTALITY]
    assert labels == {
        _UN_MORTALITY: "Under-five mortality",
        _UN_MORTALITY_FEMALE: "Under-five mortality [Sex=Female]",
        _UN_SEX: "Sex",
        _UN_FEMALE: "Female",
        _UN_SERIES: f"{_UN_SERIES} name",
    }
    assert client.node_label(_UN_FEMALE) == "Female"
    assert client.node_types_batch([_UN_MORTALITY, _UN_TOPIC, _UN_FEMALE]) == {
        _UN_MORTALITY: "StatisticalVariable",
        _UN_TOPIC: "Topic",
        _UN_FEMALE: "Enumeration",
    }
    assert client.node_type(_UN_MORTALITY) == "StatisticalVariable"
    assert client.node_observation_properties_batch(members) == dict.fromkeys(members, ())
    assert client.node_arcs_many(members, properties=["constraintProperties"]) == {
        _UN_MORTALITY: {"constraintProperties": ()},
        _UN_MORTALITY_FEMALE: {"constraintProperties": (_UN_SEX,)},
    }
    assert client.node_arc_values(_UN_MORTALITY_FEMALE, properties=[_UN_SEX, "unit"]) == {
        _UN_SEX: (_UN_FEMALE,),
        "unit": (),
    }
    assert client.node_names_many([_UN_MORTALITY]) == {_UN_MORTALITY: ("Under-five mortality",)}
    assert client.exists(_UN_MORTALITY)
    assert client.exists_batch([*members, _UN_AID, _UN_TOPIC]) == {*members, _UN_TOPIC}
    assert graph.on(_UN_HOST) == []


@pytest.mark.parametrize(
    ("filters", "max_results", "expected"),
    [
        pytest.param({}, 10, ((_UN_MORTALITY, _UN_MORTALITY_FEMALE), False), id="whole_series"),
        pytest.param(
            {_UN_SEX: _UN_FEMALE}, 10, ((_UN_MORTALITY_FEMALE,), False), id="bound_breakdown"
        ),
        pytest.param({_UN_SEX: "undata/SEX-M"}, 10, ((), False), id="value_no_member_holds"),
        pytest.param({"statType": "medianValue"}, 10, ((), False), id="other_stat_type"),
        pytest.param({}, 1, ((_UN_MORTALITY,), True), id="past_the_cap"),
        pytest.param({}, 2, ((_UN_MORTALITY, _UN_MORTALITY_FEMALE), False), id="at_the_cap"),
    ],
)
def test_a_un_series_subjects_come_from_the_un_tables_with_no_request(
    graph: _Graph,
    client: OriginGraphClient,
    bind_catalogue,
    filters: dict[str, str],
    max_results: int,
    expected: tuple[tuple[str, ...], bool],
) -> None:
    """The series decides the read whether it is the lead or a filter value."""
    bind_catalogue(_catalogue())

    as_filter = client.node_subjects(
        "value",
        property="measuredProperty",
        filters={"populationType": _UN_SERIES, **filters},
        max_results=max_results,
    )
    as_lead = client.node_subjects(
        _UN_SERIES,
        property="populationType",
        filters={"measuredProperty": "value", **filters},
        max_results=max_results,
    )

    assert (as_filter.dcids, as_filter.truncated) == expected
    assert as_lead == as_filter
    assert graph.requests == []


def test_an_excluded_un_series_has_no_subjects_and_sends_no_request(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())

    page = client.node_subjects(
        "value",
        property="measuredProperty",
        filters={"populationType": _UN_AID_SERIES},
        max_results=10,
    )

    assert (page.dcids, page.truncated) == ((), False)
    assert graph.requests == []


def test_subject_reads_naming_no_series_the_un_tables_hold_route_as_before(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    """A base series goes to our mixer, and a UN name beside no held series to the UN."""
    bind_catalogue(_catalogue())

    client.node_subjects(
        "count", property="measuredProperty", filters={"populationType": "Person"}, max_results=10
    )
    assert [json.loads(r.content)["nodes"] for r in graph.on(_OURS_HOST)] == [["count"]]
    assert graph.on(_UN_HOST) == []

    client.node_subjects(
        "value",
        property="measuredProperty",
        filters={"populationType": _UN_DEBT_SERIES, _UN_SEX: _UN_FEMALE},
        max_results=10,
    )
    client.node_subjects(_UN_FEMALE, property=_UN_SEX, max_results=10)
    assert len(graph.on(_UN_HOST)) == 2
    assert len(graph.on(_OURS_HOST)) == 1


def test_a_batch_of_series_leads_reads_only_the_ones_the_un_tables_lack(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())

    pages = client.node_subjects_many(
        [_UN_SERIES, "Person", _UN_AID_SERIES], property="populationType", max_results=1
    )

    assert pages == {
        _UN_SERIES: ((_UN_MORTALITY,), "capped"),
        "Person": ((), "complete"),
        _UN_AID_SERIES: ((), "complete"),
    }
    assert graph.dcids_sent(_OURS_HOST) == ["Person"]
    assert graph.on(_UN_HOST) == []


def test_a_un_topic_walks_to_its_table_leaves_with_no_un_request(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())

    walked = walk_topics([_UN_TOPIC], graph=client)

    assert walked == {_UN_TOPIC: [_UN_MORTALITY, _UN_MORTALITY_FEMALE]}
    assert graph.on(_UN_HOST) == []


def test_an_excluded_un_variable_reads_as_an_absent_node_and_holds_no_data(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    graph.labels[_UN_AID] = "Aid for health"
    graph.facets[(_UN_AID, "country/KEN")] = [_UN_WHO]
    bind_catalogue(_catalogue())

    assert client.node_arcs_batch([_UN_AID]) == {_UN_AID: None}
    assert client.node_labels_batch([_UN_AID]) == {}
    assert client.node_arcs_many([_UN_AID], properties=["name"]) == {_UN_AID: None}
    assert client.node_observation_properties_batch([_UN_AID]) == {_UN_AID: ()}
    assert not client.exists(_UN_AID)
    assert client.observation_facets_batch([_UN_AID], ["country/KEN"]) == {
        _UN_AID: {"country/KEN": []}
    }
    assert client.entities_with_data([_UN_AID], ["country/KEN"]) == []
    assert client.entities_with_data_by_variable([_UN_AID], ["country/KEN"]) == {_UN_AID: []}
    assert graph.on(_UN_HOST) == []


def test_withheld_at_names_every_provenance_holding_an_excluded_un_variable(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    graph.facets[(_UN_AID, "country/KEN")] = [_UN_WHO]
    bind_catalogue(_catalogue())

    assert client.withheld_at([_UN_AID], ["country/KEN", "country/USA"]) == {
        "country/KEN": (_UN_WHO,)
    }


def test_a_un_dcid_outside_the_un_tables_is_read_live_as_a_node_and_never_for_data(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    unknown = "undata/who/NOT_IN_THE_BUILD"
    graph.labels[_UN_UNIT] = "Percent"
    graph.facets[(unknown, "country/KEN")] = [_UN_WHO]
    bind_catalogue(_catalogue())

    assert client.node_labels_batch([_UN_UNIT]) == {_UN_UNIT: "Percent"}
    assert graph.dcids_sent(_UN_HOST) == [_UN_UNIT]
    assert client.observation_facets_batch([unknown], ["country/KEN"]) == {
        unknown: {"country/KEN": []}
    }
    assert client.withheld_at([unknown], ["country/KEN"]) == {}
    assert graph.observation_requests(_UN_HOST) == []


def test_with_the_un_off_the_un_tables_answer_nothing(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue(un_system_enabled=False))

    assert client.node_arcs_batch([_UN_MORTALITY]) == {_UN_MORTALITY: None}
    assert client.node_labels_batch([_UN_MORTALITY, _UN_FEMALE]) == {}
    assert client.node_type(_UN_TOPIC) is None
    assert graph.on(_UN_HOST) == []


# -- DFO and base node facts from the arc tables --


@pytest.mark.usefixtures("released_crs")
def test_our_nodes_facts_come_from_the_arc_tables_with_no_request(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    """Table answers, the release's DFO arcs among them, reach neither backend nor its caches:
    the counters do not move."""
    bind_catalogue(_catalogue(arcs=_ARCS))
    before = client.stats()

    arcs = client.node_arcs_batch([_CRS_VARIABLE, _GDP])

    assert parse_arcs(
        arcs[_CRS_VARIABLE] or {},
        ["typeOf", "constraintProperties", "populationType", "measurementDenominator"],
    ) == {
        "typeOf": ("StatisticalVariable",),
        "constraintProperties": (_RECIPIENT,),
        "populationType": (_FLOW,),
        "measurementDenominator": (_HOUSEHOLDS,),
    }
    assert [node["dcid"] for node in arcs[_CRS_VARIABLE]["constraintProperties"]["nodes"]] == [
        _RECIPIENT
    ]
    assert client.node_arcs(_GDP) == arcs[_GDP]
    assert client.node_labels_batch([_CRS_VARIABLE, _GDP, _RECIPIENT, _FLOW]) == {
        _CRS_VARIABLE: "Gross ODA disbursements",
        _GDP: "Gross domestic product",
        _RECIPIENT: "Financing recipient",
        _FLOW: "Financial flow",
    }
    assert client.node_label(_FLOW) == "Financial flow"
    # The recipient role is held with no `typeOf` arc, so it has no type to read.
    assert client.node_types_batch(
        [_CRS_VARIABLE, _HOUSEHOLDS, _RECIPIENT, "measuredValue", _OUR_TOPIC]
    ) == {
        _CRS_VARIABLE: "StatisticalVariable",
        _HOUSEHOLDS: "StatisticalVariable",
        "measuredValue": "Property",
        _OUR_TOPIC: "Topic",
    }
    assert client.node_type(_GDP) == "StatisticalVariable"
    assert client.node_names_many([_GDP]) == {
        _GDP: ("GDP", "Gross domestic product", "Economic output")
    }
    assert client.node_observation_properties_batch([_CRS_VARIABLE, _GDP]) == {
        _CRS_VARIABLE: (_RECIPIENT,),
        _GDP: (),
    }
    assert client.node_arcs_many(
        [_RECIPIENT, "Country"], properties=["rangeIncludes", "subClassOf"]
    ) == {
        _RECIPIENT: {"rangeIncludes": ("Country",), "subClassOf": ()},
        "Country": {"rangeIncludes": (), "subClassOf": ("Place",)},
    }
    assert client.node_arc_values(_CRS_VARIABLE, properties=["measuredProperty", "unit"]) == {
        "measuredProperty": (_AMOUNT,),
        "unit": (),
    }
    assert client.exists(_HOUSEHOLDS)
    assert client.exists_batch([_CRS_VARIABLE, _GDP, "Place", _OUR_TOPIC]) == {
        _CRS_VARIABLE,
        _GDP,
        _OUR_TOPIC,
    }
    assert graph.requests == []
    assert client.stats() == before


def test_a_node_an_arc_names_takes_its_label_from_the_labels_and_keeps_its_arc_name(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    """The graph's `->*` read names a nameless node by its dcid and a node of several names
    by its first, and `node_label` gives the first no label and the second its last name.
    The release and the tables answer each read as the graph does, with no request."""
    age, alcohol = "Years0", "Alcohol"
    crs = {
        "typeOf": {"nodes": [_STATVAR]},
        "constraintProperties": {
            "nodes": [
                {"dcid": age, "name": age, "types": ["Quantity"]},
                {"dcid": alcohol, "name": "Alcohol", "types": ["Substance"]},
            ]
        },
    }
    arcs = build_arc_tables(
        dfo={_CRS_VARIABLE: crs},
        base={},
        schema={},
        labels={age: None, alcohol: "Alcoholic beverages"},
    )
    bind_catalogue(_catalogue(arcs=arcs))
    release = origin_release({_CRS_VARIABLE: crs})

    with bound_release(release):
        labels = client.node_labels_batch([age, alcohol])
        nodes = client.node_arcs_batch([_CRS_VARIABLE])[_CRS_VARIABLE] or {}

    assert labels == {alcohol: "Alcoholic beverages"}
    assert client.node_label(age) is None
    assert {node["dcid"]: node["name"] for node in nodes["constraintProperties"]["nodes"]} == {
        age: age,
        alcohol: "Alcohol",
    }
    assert graph.requests == []


def test_a_node_the_arc_tables_hold_empty_reads_as_absent_with_no_request(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    """A listed variable with no arcs is a confirmed absence, whatever the graph would say."""
    graph.labels[_POPULATION] = "Population"
    bind_catalogue(_catalogue(arcs=_ARCS))

    assert client.node_arcs_batch([_POPULATION]) == {_POPULATION: None}
    assert client.node_arcs(_POPULATION) is None
    assert client.node_arcs_many([_POPULATION], properties=["name"]) == {_POPULATION: None}
    assert client.node_arc_values(_POPULATION, properties=["name"]) == {"name": ()}
    assert client.node_labels_batch([_POPULATION]) == {}
    assert client.node_label(_POPULATION) is None
    assert client.node_types_batch([_POPULATION]) == {}
    assert client.node_names_many([_POPULATION]) == {_POPULATION: ()}
    assert client.node_observation_properties_batch([_POPULATION]) == {_POPULATION: ()}
    assert not client.exists(_POPULATION)
    assert client.exists_batch([_POPULATION]) == set()
    assert graph.requests == []


@pytest.mark.usefixtures("released_crs")
def test_a_dcid_outside_the_arc_tables_is_read_live_and_alone(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    """An unlisted variable is out of the tables' scope, not absent, so it reads from our
    mixer, and a batch sends only it. A node an arc names with several types reads its type
    live, since the arc gives them in no order the graph's `typeOf` read shares."""
    graph.labels[_UNLISTED] = "Female population"
    bind_catalogue(_catalogue(arcs=_ARCS))

    arcs = client.node_arcs_batch([_GDP, _UNLISTED, _CRS_VARIABLE])
    labels = client.node_labels_batch([_GDP, _UNLISTED])
    client.node_types_batch([_RECIPIENT, _AMOUNT])

    assert list(arcs) == [_GDP, _UNLISTED, _CRS_VARIABLE]
    assert parse_arcs(arcs[_UNLISTED] or {}, ["name"]) == {"name": ("Female population",)}
    assert labels == {_GDP: "Gross domestic product", _UNLISTED: "Female population"}
    assert graph.dcids_sent(_OURS_HOST) == [_UNLISTED, _UNLISTED, _AMOUNT]
    assert graph.on(_UN_HOST) == []


def test_our_topic_walks_to_its_table_leaves_with_no_request(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    """The walk keeps every leaf the topic table holds, those the tables hold no arcs for
    too, and its admission filter applies as on a live walk."""
    bind_catalogue(_catalogue(arcs=_ARCS))

    walked = walk_topics([_OUR_TOPIC], graph=client)
    admitted = walk_topics([_OUR_TOPIC], graph=client, admit=lambda dcid: dcid != _UNLISTED)

    assert walked == {_OUR_TOPIC: [_GDP, _POPULATION, _UNLISTED]}
    assert admitted == {_OUR_TOPIC: [_GDP, _POPULATION]}
    assert graph.requests == []


# -- admission and the domain filter --


def test_facets_are_admitted_per_place_from_every_origin(
    client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())

    facets = client.observation_facets_batch(
        [_POPULATION, _UN_MORTALITY], ["country/KEN", "country/USA", "geoId/06"]
    )

    assert _providers(facets[_POPULATION]) == {
        "country/KEN": [_CRS, _WDI],
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


def test_read_groups_start_at_each_un_request_and_keep_the_input_order(
    graph: _Graph, bind_catalogue
) -> None:
    """Under a URL budget that fits one UN variable to a GET, a read of two UN variables
    among ours is two groups, each sending one UN request, and a read naming none is one."""
    bind_catalogue(_catalogue())
    stat_vars = [_POPULATION, _UN_MORTALITY, _GDP, _UN_MORTALITY_FEMALE, _CRS_VARIABLE]

    with _client(graph, un_budget=150) as client:
        groups = client.observation_read_groups(stat_vars, ["country/KEN"])
        planned_without_a_request = not graph.requests
        for group in groups:
            client.observation_facets_batch(group, ["country/KEN"])
        ours_alone = client.observation_read_groups([_POPULATION, _GDP], ["country/KEN"])

    assert planned_without_a_request
    assert groups == [
        [_POPULATION, _UN_MORTALITY, _GDP],
        [_UN_MORTALITY_FEMALE, _CRS_VARIABLE],
    ]
    assert [
        request.url.params.get_list("variable.dcids")
        for request in graph.observation_requests(_UN_HOST)
    ] == [[_UN_MORTALITY], [_UN_MORTALITY_FEMALE]]
    assert ours_alone == [[_POPULATION, _GDP]]


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


@pytest.mark.parametrize(
    ("allowed", "withheld"),
    [
        # Every facet at KEN and USA sits outside the scope: the request left them out.
        ({_UN_WHO}, {}),
        # Census is outside the scope, so `geoId/06` withholds nothing it would name.
        ({_WDI, _CRS}, {}),
        # Census inside the scope and still not admitted is named.
        ({_WDI, _CENSUS}, {"geoId/06": (_CENSUS,)}),
    ],
)
def test_withheld_at_never_names_a_provenance_outside_the_scope(
    client: OriginGraphClient, bind_catalogue, allowed: set[str], withheld: dict
) -> None:
    bind_catalogue(_catalogue().scoped(frozenset(allowed)))

    assert client.withheld_at([_POPULATION], ["country/KEN", "country/USA", "geoId/06"]) == withheld


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


def test_presence_under_a_scope_reads_scan_places_live(
    graph: _Graph, client: OriginGraphClient, bind_catalogue
) -> None:
    """The table places `_GDP` at Kenya under WDI. A scope holding only CRS leaves that row
    out, so Kenya is read live, where `_GDP` holds no CRS facet."""
    bind_catalogue(_catalogue().scoped(frozenset({_CRS})))
    places = ["country/KEN", "geoId/06"]

    assert client.entities_with_data([_GDP], places) == []
    assert client.entities_with_data_by_variable([_GDP], places) == {_GDP: []}
    assert [
        json.loads(r.content)["entity"]["dcids"] for r in graph.observation_requests(_OURS_HOST)
    ] == [places, places]


def test_a_scope_excluding_dfo_lets_a_base_facet_serve_at_a_dfo_place(
    client: OriginGraphClient, bind_catalogue
) -> None:
    bind_catalogue(_catalogue().scoped(frozenset({_WDI, _CENSUS})))

    facets = client.observation_facets_batch([_POPULATION], ["country/KEN", "geoId/06"])

    assert _providers(facets[_POPULATION]) == {"country/KEN": [_WDI], "geoId/06": []}


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

    assert _providers(facets[_POPULATION]) == {"country/KEN": [_CRS, _WDI], "country/USA": [_WDI]}
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
    assert _providers(facets[_POPULATION]) == {"country/KEN": [_CRS, _WDI], "country/USA": [_WDI]}
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
