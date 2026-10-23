"""A variable's card on the node route and its `facets` list on the members route, read through
the surface: one profile read per response, place kinds from each place's own node, the national
currencies once per generation, and a failed read leaving a transient gap."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import contextmanager

import httpx
import pytest

from dc_sdmx import SdmxClient
from graph_api.availability import AVAILABILITY_GAP, COUNTRIES_ONLY_GAP
from graph_api.card_reads import (
    LABELS_GAP,
    NATIONAL_CURRENCY_GAP,
    PLACE_KINDS_GAPS,
    CardReadCaches,
    FactsRead,
)
from graph_api.doors import members as members_door
from graph_api.doors import node as node_door
from graph_generation import UNKNOWN_GENERATION
from graph_release.context import bound_release, current_release, served_origin_key
from materialise import EnrichmentLookup, MaterialisationRequest, materialise_spec, place_scope
from materialise.facet_profile import FacetProfile, read_facet_profiles
from materialise.graph import Facet, GraphReadError, SubjectPage
from materialise.place_scope import DEFAULT_SCOPE_CHILD_TYPE, DEFAULT_SCOPE_PARENT
from materialise.window import RequestedWindow
from origin_catalogue import DFO, Catalogue, bound_catalogue, current_catalogue
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.graph_release.support import release_at_revision
from tests.materialise.conftest import FakeGraphReader
from tests.origin_catalogue.fake_catalogue import provenance_row, restrictive_catalogue

from .conftest import FakeGraphNodeReader, make_client, wait_until

_GDP = "GDP_Nominal"
_DEFLATOR = "GDP_Deflator"
_FLOWS = "Amount_Flows"
_WEO = "provenance/IMFWorldEconomicOutlook"
_OTHER = "provenance/SomewhereElse"
_FRANCE, _GERMANY, _AFRICA = "country/FRA", "country/DEU", "africa"
_SDMX_BASE = "https://datacommons.one.org/core/api"


def _facet(facet_id: str, unit: str, method: str | None) -> Facet:
    """An annual facet at one place: reported values from 2000 to 2020, or estimates for 2019
    and 2020, one observation a year."""
    first = 2000 if method is None else 2019
    return Facet(
        facet_id=facet_id,
        provenance_id=_WEO,
        measurement_method=method,
        unit=unit,
        observation_period="P1Y",
        obs_count=2020 - first + 1,
        earliest_date=str(first),
        latest_date="2020",
    )


_USD = _facet("fUSD", "CurrencyUSD", None)
_EUR = _facet("fEUR", "CurrencyEUR", "Estimated")
_PERCENT = _facet("fPct", "Percent", None)


class _ProfileGraph(FakeGraphReader):
    """`FakeGraphReader` taking the facet read's byte cap, and calling `before_facets_read`
    before each facet read when it is set."""

    before_facets_read: Callable[[], object] | None = None

    def observation_facets_batch_many(
        self,
        *,
        stat_vars: Sequence[str],
        entities: Sequence[str],
        deadline: float | None = None,
        max_bytes: int | None = None,
    ) -> dict[str, dict[str, list[Facet]]]:
        if self.before_facets_read is not None:
            self.before_facets_read()
        return super().observation_facets_batch_many(
            stat_vars=stat_vars, entities=entities, deadline=deadline
        )


def _profile_graph() -> _ProfileGraph:
    return _ProfileGraph(
        facets={
            (_GDP, _FRANCE): [_USD, _EUR],
            (_GDP, _GERMANY): [_USD, _EUR],
            (_DEFLATOR, _FRANCE): [_PERCENT],
            (_DEFLATOR, _AFRICA): [_PERCENT],
        },
        child_places={(DEFAULT_SCOPE_PARENT, DEFAULT_SCOPE_CHILD_TYPE): [_FRANCE, _GERMANY]},
    )


_NODES: dict[str, dict[str, tuple[str, ...]]] = {
    _GDP: {"typeOf": ("StatisticalVariable",), "name": ("GDP, current prices",)},
    _DEFLATOR: {"typeOf": ("StatisticalVariable",), "name": ("GDP deflator",)},
    _FLOWS: {"typeOf": ("StatisticalVariable",), "name": ("Flows",)},
    _FRANCE: {"typeOf": ("Country",), "name": ("France",)},
    _GERMANY: {"typeOf": ("Country",), "name": ("Germany",)},
    _AFRICA: {"typeOf": ("Continent",), "name": ("Africa",)},
    "Estimated": {"typeOf": ("MeasurementMethodEnum",), "name": ("estimated",)},
    "Percent": {"typeOf": ("UnitOfMeasure",), "name": ("percent",)},
    "CurrencyEUR": {"typeOf": ("CurrencyUnit",), "nationalCurrencyOf": (_FRANCE, _GERMANY)},
    "CurrencyUSD": {"typeOf": ("CurrencyUnit",), "nationalCurrencyOf": ("country/USA",)},
}
_CURRENCY_UNITS = {
    ("CurrencyUnit", "typeOf"): SubjectPage(dcids=("CurrencyEUR", "CurrencyUSD"), truncated=False)
}


_FACTS = ("name", "typeOf")


class _Reads(FakeGraphNodeReader):
    """Records each name and type read, and fails one that names a dcid in `fail_facts`, or
    the national-currency listing when `fail_currencies` is set. With `release`, a facts read
    waits for it, or only one naming a dcid in `hold` when that is set."""

    def __init__(
        self,
        *,
        fail_facts: frozenset[str] = frozenset(),
        fail_currencies: bool = False,
        release: threading.Event | None = None,
        hold: frozenset[str] | None = None,
    ) -> None:
        super().__init__(nodes=_NODES, subjects=_CURRENCY_UNITS)
        self.fail_facts = fail_facts
        self._fail_currencies = fail_currencies
        self._release = release
        self._hold = hold
        self.fact_reads: list[tuple[str, ...]] = []
        self.facts_landed = threading.Event()

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        if tuple(properties) != _FACTS:
            return super().node_arcs_many(dcids, properties=properties, deadline=deadline)
        self.fact_reads.append(tuple(sorted(dcids)))
        if self._release is not None and (self._hold is None or self._hold.intersection(dcids)):
            assert self._release.wait(timeout=5.0), "the test never released the facts read"
        if self.fail_facts.intersection(dcids):
            raise GraphReadError("names and types unavailable")
        found = super().node_arcs_many(dcids, properties=properties, deadline=deadline)
        self.facts_landed.set()
        return found

    def node_subjects(self, dcid, *, property, filters=None, max_results, deadline=None):
        if self._fail_currencies:
            raise GraphReadError("currency listing unavailable")
        return super().node_subjects(
            dcid, property=property, filters=filters, max_results=max_results, deadline=deadline
        )


@pytest.fixture
def weo_catalogue(serve_catalogue) -> Catalogue:
    catalogue = restrictive_catalogue(
        provenances=[provenance_row(_WEO, DFO)], dfo_variables=[_GDP, _DEFLATOR, _FLOWS]
    )
    serve_catalogue(catalogue)
    return catalogue


def _coverage() -> FakeCoverageService:
    """A role-bearing variable with one facet in percent, between two parties."""
    return FakeCoverageService(
        build_coverage_index(
            [(_FLOWS, "fFlows", "donor/A", "recipient/B", [2010, 2011])],
            role_names={
                _FLOWS: RoleNames(donor="financingProvider", recipient="financingRecipient")
            },
            facet_metadata={
                "fFlows": FacetMetadata(
                    provenance_id=_WEO, unit="Percent", observation_period="P1Y"
                )
            },
        )
    )


def _client(
    graph: FakeGraphNodeReader | None = None,
    profile_graph: _ProfileGraph | None = None,
    node_pool_size: int | None = None,
    coverage: FakeCoverageService | None = None,
):
    profile_graph = profile_graph or _profile_graph()
    return make_client(
        graph=graph or _Reads(),
        coverage=coverage or _coverage(),
        profile_graph=profile_graph,
        availability=profile_graph.scope_availability(),
        node_pool_size=node_pool_size,
    )


@pytest.mark.usefixtures("weo_catalogue")
class TestVariableCard:
    def test_a_card_folds_its_availability_and_names_the_builds_it_read(self, weo_catalogue):
        profile_graph = _profile_graph()
        with bound_catalogue(weo_catalogue):
            elected = read_facet_profiles(
                [_GDP],
                graph=profile_graph,
                availability=profile_graph.scope_availability(),
                coverage=_coverage().snapshot().index,
                max_bytes=1 << 20,
                deadline=None,
            )[_GDP]
        with _client() as client:
            resp = client.get(f"/api/graph/node/{_GDP}")

        assert resp.status_code == 200, resp.text
        assert resp.headers["cache-control"] == "public, max-age=300"
        body = resp.json()
        assert body["built_from"]["coverage_index"] == "test-coverage-stamp-build"
        assert "origin_catalogue" in body["built_from"]
        assert not [gap for gap in body["gaps"] if gap.startswith("stat_var.availability")]
        availability = body["stat_var"]["availability"]
        assert availability["summary"] == {
            "places": {"country": 2},
            "entities": None,
            "first_year": "2000",
            "last_year": "2020",
            "sources": 1,
            "facets": 2,
        }
        assert isinstance(elected, FacetProfile)
        # The reported series holds more observations, so it leads a request naming nothing.
        assert availability["default"]["facets"] == [f.facet_id for f in elected.default]
        assert availability["default"]["facets"] == ["fUSD"]
        [source] = availability["sources"]
        by_method = {
            (series["measurement_method"] or {}).get("label"): series for series in source["series"]
        }
        assert set(by_method) == {None, "estimated"}
        assert by_method["estimated"]["units"]["national_currency"] == {
            "currencies": 1,
            "places": 2,
        }
        assert by_method["estimated"]["units"]["leading"][0]["label"] == "euros"
        assert by_method[None]["units"]["national_currency"] is None

    def test_a_place_counts_under_the_kind_its_own_node_reports(self):
        with _client() as client:
            resp = client.get(f"/api/graph/node/{_DEFLATOR}")

        availability = resp.json()["stat_var"]["availability"]
        assert availability["summary"]["places"] == {"continent": 1, "country": 1}
        unit = availability["sources"][0]["series"][0]["units"]["leading"][0]
        assert unit == {"ref": "Percent", "label": "percent", "places": 2}

    def test_a_batch_reads_each_variables_facets_once_in_its_own_flight(self):
        profile_graph = _profile_graph()
        with _client(profile_graph=profile_graph) as client:
            resp = client.post("/api/graph/nodes", json={"dcids": [_GDP, _DEFLATOR]})

        assert resp.status_code == 200, resp.text
        assert [
            len(node["stat_var"]["availability"]["sources"]) for node in resp.json()["nodes"]
        ] == [
            1,
            1,
        ]
        assert profile_graph.calls["observation_facets_batch_many"] == 2

    def test_a_second_card_reads_no_currency_name_or_place_again(self):
        graph = _Reads()
        with _client(graph=graph) as client:
            first = client.get(f"/api/graph/node/{_GDP}")
            second = client.get(f"/api/graph/node/{_GDP}")
            third = client.get(f"/api/graph/node/{_DEFLATOR}")

        assert first.json()["stat_var"] == second.json()["stat_var"]
        assert third.status_code == 200, third.text
        assert graph.calls["node_subjects"] == 1
        # The deflator's card reads only the facts the first card left unread.
        assert graph.fact_reads == [
            ("Estimated", _GERMANY, _FRANCE),
            ("Percent", _AFRICA),
        ]

    @pytest.mark.parametrize(
        ("stat_var", "failing", "gaps"),
        [
            pytest.param(
                _GDP,
                {"fail_facts": frozenset({"Estimated"})},
                [LABELS_GAP, *PLACE_KINDS_GAPS],
                id="names-and-place-types",
            ),
            pytest.param(
                _FLOWS,
                {"fail_facts": frozenset({"Percent"})},
                [LABELS_GAP],
                id="names-of-a-role-bearing-card",
            ),
            pytest.param(_GDP, {"fail_currencies": True}, [NATIONAL_CURRENCY_GAP], id="currencies"),
        ],
    )
    def test_a_failed_input_read_folds_the_card_with_a_transient_gap(self, stat_var, failing, gaps):
        with _client(graph=_Reads(**failing)) as client:
            resp = client.get(f"/api/graph/node/{stat_var}")

        assert resp.status_code == 200, resp.text
        assert resp.headers["cache-control"] == "no-store"
        body = resp.json()
        assert [gap for gap in body["gaps"] if gap.startswith("stat_var.availability")] == gaps
        availability = body["stat_var"]["availability"]
        [source] = availability["sources"]
        units = [series["units"] for series in source["series"]]
        if PLACE_KINDS_GAPS[0] in gaps:
            assert (availability["summary"]["places"], source["places"]) == (None, None)
            assert availability["default"]["places"] == 2
        if NATIONAL_CURRENCY_GAP in gaps:
            assert [fold["national_currency"] for fold in units] == [None, None]
        if LABELS_GAP in gaps:
            named = [series["measurement_method"] for series in source["series"]]
            named += [
                fold["leading"][0] for fold in units if fold["leading"][0]["ref"] == "Percent"
            ]
            assert [ref["label"] for ref in named if ref is not None] == [None]

    def test_a_currency_read_the_dcp_executor_refuses_folds_the_card_with_a_transient_gap(self):
        profile_graph = _profile_graph()
        held = threading.Event()
        with _client(profile_graph=profile_graph, node_pool_size=1) as client:
            dcp = client.app.state.graph_deps.dcp_executor
            # The node read has returned its worker; hold it until the card's inputs are read.
            profile_graph.before_facets_read = lambda: dcp.submit(held.wait, 5.0)
            try:
                resp = client.get(f"/api/graph/node/{_GDP}")
            finally:
                held.set()

        assert resp.status_code == 200, resp.text
        assert resp.headers["cache-control"] == "no-store"
        assert NATIONAL_CURRENCY_GAP in resp.json()["gaps"]
        [source] = resp.json()["stat_var"]["availability"]["sources"]
        assert [series["units"]["national_currency"] for series in source["series"]] == [None, None]

    def test_a_facts_read_past_the_deadline_fills_the_cache_for_the_next_card(self, monkeypatch):
        monkeypatch.setattr(node_door, "GRAPH_NODE_TIMEOUT_SECONDS", 1)
        release = threading.Event()
        graph = _Reads(release=release)
        with _client(graph=graph) as client:
            first = client.get(f"/api/graph/node/{_GDP}")
            release.set()
            assert graph.facts_landed.wait(timeout=5.0)
            reads = list(graph.fact_reads)
            wait_until(lambda: _card_places(client) is not None)
            second = client.get(f"/api/graph/node/{_GDP}")

        assert first.status_code == 200, first.text
        assert first.headers["cache-control"] == "no-store"
        first_gaps = [g for g in first.json()["gaps"] if g.startswith("stat_var.availability")]
        assert first_gaps == [LABELS_GAP, *PLACE_KINDS_GAPS]
        assert first.json()["stat_var"]["availability"]["summary"]["places"] is None
        assert second.json()["stat_var"]["availability"]["summary"]["places"] == {"country": 2}
        assert second.headers["cache-control"] == "public, max-age=300"
        assert graph.fact_reads == reads == [("Estimated", _GERMANY, _FRANCE)]

    def test_a_profile_miss_gives_a_null_availability_with_a_transient_gap(self):
        profile_graph = _profile_graph()

        def fail() -> None:
            raise GraphReadError("facets unavailable")

        profile_graph.before_facets_read = fail
        with _client(profile_graph=profile_graph) as client:
            resp = client.get(f"/api/graph/node/{_GDP}")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["stat_var"]["availability"] is None
        assert AVAILABILITY_GAP in body["gaps"]
        assert resp.headers["cache-control"] == "no-store"


def _two_facet_coverage() -> FakeCoverageService:
    """The role-bearing variable with a facet in percent between two pairs of parties, and one
    in US dollars between one pair over more years."""
    return FakeCoverageService(
        build_coverage_index(
            [
                (_FLOWS, "fFlows", "donor/A", "recipient/B", [2010, 2011]),
                (_FLOWS, "fFlows", "donor/A", "recipient/C", [2010, 2011]),
                (_FLOWS, "fFlowsUSD", "donor/A", "recipient/B", [2008, 2009, 2010, 2011, 2012]),
            ],
            role_names={
                _FLOWS: RoleNames(donor="financingProvider", recipient="financingRecipient")
            },
            facet_metadata={
                "fFlows": FacetMetadata(
                    provenance_id=_WEO, unit="Percent", observation_period="P1Y"
                ),
                "fFlowsUSD": FacetMetadata(
                    provenance_id=_WEO, unit="CurrencyUSD", observation_period="P1Y"
                ),
            },
        )
    )


def _mint_dispatches(
    stat_var: str,
    declared_roles: tuple[str, ...],
    *,
    profile_graph: _ProfileGraph,
    coverage: FakeCoverageService,
) -> set[str]:
    """The facet ids a mint of `stat_var` binding no place, year or facet dispatches, over the
    card's graph, availability and coverage. Each data read answers no rows."""
    availability = profile_graph.scope_availability()

    header = (
        f"STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,{','.join(declared_roles)},unit,"
        "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,"
        "facetId\r\n"
    )

    def answer(request: httpx.Request) -> httpx.Response:
        if (response := availability.respond(request)) is not None:
            return response
        return httpx.Response(
            200, text=header, headers={"content-type": "application/vnd.sdmx.data+csv"}
        )

    result = materialise_spec(
        MaterialisationRequest(
            stat_var=f"dcid:{stat_var}",
            role_bindings={},
            declared_roles=declared_roles,
            facet_bindings={},
            window=RequestedWindow(kind="all"),
        ),
        sdmx=SdmxClient(base_url=_SDMX_BASE, transport=httpx.MockTransport(answer)),
        graph=profile_graph,
        enrichment=EnrichmentLookup(profile_graph, parties=lambda: {}),
        materialiser_build="build",
        capability_version="cap",
        coverage=coverage.snapshot().index,
    )
    return {str(entry["dispatch_facet_id"]) for entry in result.trace.facets}


@pytest.mark.parametrize(
    ("stat_var", "declared_roles"),
    [
        pytest.param(_GDP, ("observationAbout",), id="subject-only"),
        pytest.param(_FLOWS, ("financingProvider", "financingRecipient"), id="role-bearing"),
    ],
)
def test_a_cards_default_facets_are_the_ones_a_no_place_mint_dispatches(
    weo_catalogue, stat_var, declared_roles
):
    with _client(coverage=_two_facet_coverage()) as client:
        card = client.get(f"/api/graph/node/{stat_var}").json()
    with bound_catalogue(weo_catalogue):
        dispatched = _mint_dispatches(
            stat_var,
            declared_roles,
            profile_graph=_profile_graph(),
            coverage=_two_facet_coverage(),
        )

    default = card["stat_var"]["availability"]["default"]["facets"]
    assert dispatched
    assert set(default) == dispatched


def _card_places(client) -> dict | None:
    return client.get(f"/api/graph/node/{_GDP}").json()["stat_var"]["availability"]["summary"][
        "places"
    ]


def _facets(client, dcid: str = _GDP, **query: str):
    return client.get(f"/api/graph/node/{dcid}/members", params={"relation": "facets", **query})


@pytest.mark.usefixtures("weo_catalogue")
class TestFacetsRelation:
    @pytest.mark.parametrize(
        ("query", "refs"),
        [
            pytest.param({}, ["fEUR", "fUSD"], id="open"),
            pytest.param({"unit": "CurrencyEUR"}, ["fEUR"], id="unit"),
            pytest.param({"measurement_method": "Estimated"}, ["fEUR"], id="method"),
            pytest.param({"measurement_method": ""}, ["fUSD"], id="no-stated-method"),
            pytest.param({"source": _WEO, "unit": "CurrencyUSD"}, ["fUSD"], id="source-and-unit"),
            pytest.param({"source": _OTHER}, [], id="another-source"),
        ],
    )
    def test_a_variables_facets_filter_and_order_by_facet_id(self, query, refs):
        with _client() as client:
            resp = _facets(client, **query)

        assert resp.status_code == 200, resp.text
        page = resp.json()
        assert [item["ref"] for item in page["items"]] == refs
        assert (page["complete"], page["total"], page["gaps"]) == (True, len(refs), [])
        assert page["built_from"]["coverage_index"] == "test-coverage-stamp-build"
        assert resp.headers["cache-control"] == "public, max-age=300"

    def test_a_facet_item_carries_its_source_unit_method_and_span(self):
        with _client() as client:
            [item] = _facets(client, unit="CurrencyEUR").json()["items"]

        assert item == {
            "ref": "fEUR",
            "label": None,
            "source": {
                "ref": _WEO,
                "label": item["source"]["label"],
                "documentation": f"/api/graph/documentation/{_WEO}",
            },
            "unit": {"ref": "CurrencyEUR", "label": "euros"},
            "measurement_method": {"ref": "Estimated", "label": "estimated"},
            "observation_period": "P1Y",
            "scaling": None,
            "first_year": "2019",
            "last_year": "2020",
            "observations": 4,
            "places": 2,
        }

    def test_the_cursor_walks_the_facets_in_order(self):
        with _client() as client:
            first = _facets(client, limit="1").json()
            second = _facets(client, limit="1", cursor=first["next_cursor"]).json()

        assert [item["ref"] for item in first["items"] + second["items"]] == ["fEUR", "fUSD"]
        assert second["next_cursor"] is None

    @pytest.mark.parametrize(
        ("dcid", "status", "code"),
        [
            pytest.param(_FRANCE, 400, "request_invalid", id="held-non-variable"),
            pytest.param("dc/Unheld", 404, "node_not_held", id="unheld"),
        ],
    )
    def test_a_node_that_is_no_variable_has_no_facets(self, dcid, status, code):
        with _client() as client:
            resp = _facets(client, dcid=dcid)

        assert resp.status_code == status, resp.text
        assert resp.json()["code"] == code

    @pytest.mark.parametrize(
        "other",
        [
            pytest.param({"unit": "CurrencyUSD"}, id="unit"),
            pytest.param({"measurement_method": ""}, id="no-stated-method"),
            pytest.param({"measurement_method": "Estimated"}, id="method"),
            pytest.param({"source": _WEO}, id="source"),
        ],
    )
    def test_a_cursor_sent_with_other_filters_is_refused(self, other):
        with _client() as client:
            first = _facets(client, limit="1").json()
            resp = _facets(client, limit="1", cursor=first["next_cursor"], **other)

        assert resp.status_code == 400, resp.text
        assert resp.json()["pointer"] == "/cursor"

    @pytest.mark.parametrize("name", ["source", "unit", "measurement_method"])
    def test_another_relation_refuses_a_facet_filter(self, name):
        with _client() as client:
            resp = client.get(
                f"/api/graph/node/{_WEO}/members", params={"relation": "variables", name: "x"}
            )

        assert resp.status_code == 400, resp.text
        assert resp.json()["pointer"] == f"/{name}"

    def test_a_countries_only_profile_carries_the_cards_permanent_gap(self, monkeypatch):
        # Past the every-place cap, the default scope falls back to the countries with data.
        monkeypatch.setattr(place_scope, "MAX_EVERY_PLACE", 1)
        with _client() as client:
            card = client.get(f"/api/graph/node/{_GDP}")
            resp = _facets(client)

        assert COUNTRIES_ONLY_GAP in card.json()["gaps"]
        assert resp.status_code == 200, resp.text
        page = resp.json()
        assert [item["ref"] for item in page["items"]] == ["fEUR", "fUSD"]
        assert (page["complete"], page["gaps"]) == (True, [COUNTRIES_ONLY_GAP])
        assert resp.headers["cache-control"] == "public, max-age=300"

    def test_a_surface_that_reads_no_profiles_lists_nothing_and_is_cached(self):
        with make_client(graph=FakeGraphNodeReader(nodes=_NODES)) as client:
            resp = _facets(client)

        assert resp.status_code == 200, resp.text
        page = resp.json()
        assert (page["items"], page["complete"], page["total"]) == ([], False, None)
        assert page["gaps"] == ["facets_not_served"]
        assert resp.headers["cache-control"] == "public, max-age=300"

    def test_a_failed_label_read_leaves_labels_null_and_is_not_cached(self):
        with _client(graph=_Reads(fail_facts=frozenset({"Estimated"}))) as client:
            resp = _facets(client)

        assert resp.status_code == 200, resp.text
        page = resp.json()
        assert [item["ref"] for item in page["items"]] == ["fEUR", "fUSD"]
        assert page["items"][0]["measurement_method"] == {"ref": "Estimated", "label": None}
        assert page["gaps"] == ["labels_not_read"]
        assert resp.headers["cache-control"] == "no-store"

    def test_the_variable_check_takes_no_driver_while_every_driver_is_held(self, monkeypatch):
        """Two cold cards leave both facts drivers reading; the relation still learns its node
        is a variable, from one read on the DCP-read executor, and lists the facets."""
        monkeypatch.setattr(node_door, "GRAPH_NODE_TIMEOUT_SECONDS", 1)
        monkeypatch.setattr(members_door, "GRAPH_NODE_TIMEOUT_SECONDS", 1)
        release = threading.Event()
        graph = _Reads(release=release, hold=frozenset({_FRANCE, _AFRICA}))
        with _client(graph=graph) as client:
            try:
                for stat_var in (_GDP, _DEFLATOR):
                    client.get(f"/api/graph/node/{stat_var}")
                resp = _facets(client)
            finally:
                release.set()

        assert resp.status_code == 200, resp.text
        assert [item["ref"] for item in resp.json()["items"]] == ["fEUR", "fUSD"]

    def test_a_failed_variable_check_refuses_retryably(self):
        with _client(graph=_Reads(fail_facts=frozenset({_GDP}))) as client:
            resp = _facets(client)

        assert resp.status_code == 503, resp.text


@contextmanager
def _caches(
    graph: FakeGraphNodeReader, dcp: Executor, *, chunk_size: int, drivers: int = 1
) -> Iterator[CardReadCaches]:
    caches = CardReadCaches(
        graph=graph,
        executor=dcp,
        drivers=drivers,
        generation=UNKNOWN_GENERATION,
        maxsize=100,
        chunk_size=chunk_size,
        chunk_timeout_s=5.0,
        detached_timeout_s=5.0,
    )
    try:
        yield caches
    finally:
        caches.close()


def test_facts_land_chunk_by_chunk_and_a_later_read_takes_only_the_missing():
    graph = _Reads(fail_facts=frozenset({_AFRICA}))
    wanted = [_FRANCE, _GERMANY, _AFRICA, "Percent"]

    async def read(caches: CardReadCaches) -> tuple[FactsRead, FactsRead]:
        first = await caches.node_facts(wanted, deadline=time.monotonic() + 5.0)
        graph.fail_facts = frozenset()
        return first, await caches.node_facts(wanted, deadline=time.monotonic() + 5.0)

    with ThreadPoolExecutor(max_workers=2) as dcp, _caches(graph, dcp, chunk_size=2) as caches:
        first, second = asyncio.run(read(caches))

    assert (sorted(first.facts), first.complete) == ([_GERMANY, _FRANCE], False)
    assert (sorted(second.facts), second.complete) == (sorted(wanted), True)
    assert second.facts[_AFRICA].types == ("Continent",)
    assert graph.fact_reads == [
        (_GERMANY, _FRANCE),
        ("Percent", _AFRICA),
        ("Percent", _AFRICA),
    ]


class _ReleaseNames(_Reads):
    """Names each node after the release bound to the read, as the release answers a DFO
    variable's name."""

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        if tuple(properties) != _FACTS:
            return super().node_arcs_many(dcids, properties=properties, deadline=deadline)
        self.fact_reads.append(tuple(sorted(dcids)))
        release = current_release().release_id
        return {dcid: {"name": (f"{dcid} at {release}",), "typeOf": ("Thing",)} for dcid in dcids}


def test_facts_read_under_one_release_are_read_again_under_the_next():
    first, second = release_at_revision("rev-1"), release_at_revision("rev-2")
    graph = _ReleaseNames()

    async def read(caches: CardReadCaches) -> list[str | None]:
        names = []
        for release in (first, second, first):
            with bound_release(release):
                found = await caches.node_facts([_FRANCE], deadline=time.monotonic() + 5.0)
            names.append(found.facts[_FRANCE].name)
        return names

    with ThreadPoolExecutor(max_workers=2) as dcp, _caches(graph, dcp, chunk_size=2) as caches:
        names = asyncio.run(read(caches))

    assert names == [f"{_FRANCE} at {release.release_id}" for release in (first, second, first)]
    assert len(graph.fact_reads) == 2


def test_the_driver_reads_every_chunk_after_its_caller_stops_waiting():
    release = threading.Event()
    graph = _Reads(release=release)
    wanted = [_FRANCE, _GERMANY]

    async def read(caches: CardReadCaches) -> tuple[FactsRead, FactsRead]:
        first = await caches.node_facts(wanted, deadline=time.monotonic() + 0.1)
        release.set()
        await asyncio.to_thread(wait_until, lambda: len(graph.fact_reads) == 2)
        return first, await caches.node_facts(wanted, deadline=time.monotonic() + 5.0)

    with ThreadPoolExecutor(max_workers=2) as dcp, _caches(graph, dcp, chunk_size=1) as caches:
        first, second = asyncio.run(read(caches))

    assert (first.facts, first.complete) == ({}, False)
    assert second.complete
    assert graph.fact_reads == [(_FRANCE,), (_GERMANY,)]


def test_a_cached_read_returns_at_once_while_every_driver_is_held():
    release = threading.Event()
    graph = _Reads(release=release, hold=frozenset({_GERMANY, _AFRICA, "Percent"}))

    async def read(caches: CardReadCaches) -> tuple[FactsRead, FactsRead, float, FactsRead, int]:
        warm = await caches.node_facts([_FRANCE], deadline=time.monotonic() + 5.0)
        for held in (_GERMANY, _AFRICA):
            await caches.node_facts([held], deadline=time.monotonic() + 0.05)
        await asyncio.to_thread(wait_until, lambda: len(graph.fact_reads) == 3)
        started = time.monotonic()
        cached = await caches.node_facts([_FRANCE], deadline=started + 2.0)
        waited = time.monotonic() - started
        cold = await caches.node_facts(["Percent"], deadline=time.monotonic() + 0.1)
        return warm, cached, waited, cold, caches._drivers._work_queue.qsize()

    with (
        ThreadPoolExecutor(max_workers=4) as dcp,
        _caches(graph, dcp, chunk_size=1, drivers=2) as caches,
    ):
        try:
            warm, cached, waited, cold, queued = asyncio.run(read(caches))
        finally:
            release.set()

    assert warm.complete
    assert (list(cached.facts), cached.complete, waited < 1.0) == ([_FRANCE], True, True)
    # The cold read found no free driver by its deadline, so it left none queued to read later.
    assert (cold.facts, cold.complete, queued) == ({}, False, 0)
    assert graph.fact_reads == [(_FRANCE,), (_GERMANY,), (_AFRICA,)]


class _CatalogueReads(_Reads):
    """`_Reads` whose facts read needs the request's origin catalogue bound, as the
    catalogue-aware graph reader's does."""

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        current_catalogue()
        return super().node_arcs_many(dcids, properties=properties, deadline=deadline)


def test_the_driver_reads_under_the_callers_bound_catalogue():
    graph = _CatalogueReads()

    async def read(caches: CardReadCaches) -> FactsRead:
        with bound_catalogue(restrictive_catalogue(provenances=[provenance_row(_WEO, DFO)])):
            return await caches.node_facts([_FRANCE], deadline=time.monotonic() + 5.0)

    with ThreadPoolExecutor(max_workers=2) as dcp, _caches(graph, dcp, chunk_size=1) as caches:
        read_facts = asyncio.run(read(caches))
        key = (_FRANCE, served_origin_key())
        cached = caches._facts.cached_many([key])

    assert read_facts.complete
    assert read_facts.facts[_FRANCE].types == ("Country",)
    assert cached == {key: read_facts.facts[_FRANCE]}


def test_a_read_past_its_deadline_starts_no_driver_though_a_slot_is_free():
    graph = _Reads()

    async def read(caches: CardReadCaches) -> tuple[FactsRead, bool]:
        expired = await caches.node_facts([_FRANCE], deadline=time.monotonic() - 1.0)
        return expired, caches._driver_slots.locked()

    with ThreadPoolExecutor(max_workers=2) as dcp, _caches(graph, dcp, chunk_size=1) as caches:
        expired, held = asyncio.run(read(caches))

    assert (expired.facts, expired.complete, held) == ({}, False, False)
    assert graph.fact_reads == []
