"""Each candidate binds the query's places to its own shape's roles.

The elected shape, every alternate and a re-elected sibling bind through
`place_roles.bind_place_roles`. A subject-only alternate beside a role-bearing elected shape
binds the places on `observationAbout`; a role-bearing alternate beside a subject-only one
binds them on its one required role; a subject-only sibling re-elected from a role-bearing
shape with no data at the place answers under its own roles and probes its own places; an
alternate whose roles the places cannot bind is dropped. Each candidate is materialised,
grounded and probed over its own place context, and reports the place warnings that differ
from the shape in play's, stamped with its spec id. The fixtures mirror the dev graph's
"Gross national income (current US$) of Ghana": the IDS series declares
`financingRecipient`, its by-creditor twin also declares `financingProvider`, and the World
Bank PPP series declares no role.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import logging
import threading
import time
from collections.abc import Callable, Collection, Coroutine, Sequence
from datetime import date
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from code_index.shared import get_index
from dfo_platform.trace import reset_trace_id, set_trace_id
from entity_resolution import Entity, Resolved
from graph_generation import UNKNOWN_GENERATION
from materialise.graph import Facet as MaterialiseFacet
from materialise.graph import SubjectPage
from search.engine.bind import SlotBindingDraft
from search.engine.core import resolve_async
from search.engine.degraded_reads import recording_degraded_reads
from search.engine.errors import GraphInfraError
from search.engine.extract import Extraction
from search.engine.interpret import Recall
from search.engine.quantity_read import QuantityOutput
from search.engine.region import (
    candidate_admission,
    candidate_collapse,
    candidate_options,
    candidates,
    phase_place_materialise,
    phase_rank_bind,
    pipeline,
    place_roles,
)
from search.engine.region.candidates import _AlternatesRead
from search.engine.region.decisions import PartyReadingWithdrawn
from search.engine.region.phase_intake import _party_word_sets
from search.engine.region.phase_place_materialise import _phase_place_bindings
from search.engine.region.phase_rank_bind import (
    _dataset_stated_words,
    with_party_reading,
)
from search.engine.region.pipeline import resolve_variable
from search.engine.region.place_roles import (
    bind_place_roles,
    held_party_words,
    lacks_query_axis,
    stated_party_words,
    subject_shapes_refused,
    where_binding,
)
from search.engine.region.place_set import PlaceNode, PlaceSet
from search.engine.region.state import ResolveState
from search.engine.response_cache import ResponseCache
from search.engine.retrieval.derive import derive_shapes
from search.engine.retrieval.read_plan import PairPlan, ReadPlan
from search.engine.role_text import DoorRole
from search.engine.shape import ShapeDraft
from search.models import (
    DatasetResubmitInput,
    Operation,
    OptionsFor,
    OptionsRequest,
    RawTextInput,
    ResolveRequest,
    ResolveResponse,
)
from series_coverage.schema import RoleNames
from tests.engine import test_first_answer_place_scenarios as scenarios
from tests.engine._binders import MenuRecordingBinder, elected_reply
from tests.engine._builders import place_sets, settled_place_roles
from tests.engine._fake_coverage import (
    FakeCoverageService,
    build_coverage_index,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query, offline_resolve
from tests.engine._quantity import NO_QUANTITY
from tests.engine._search_app import search_app
from tests.engine._virtual_clock import VirtualRequests
from tests.engine.test_dataset_resubmit import _identity_entry_from_spec
from tests.fixtures import PINNED_DATE, FakeGraph, FakeLLM, make_resolver
from tests.materialise._scope_availability import ScopeAvailability
from tests.materialise.conftest import FakeGraphReader

_GHA = "country/GHA"
_NOR = "country/NOR"
_GNI_IDS = "Amount_EconomicActivity_GrossNationalIncome_Nominal"
_GNI_PPP = "Amount_EconomicActivity_GrossNationalIncome_PurchasingPowerParity"
_YEARS = [2020, 2021, 2022]


def _economic_activity_arcs(
    qualifier: str, constraints: dict[str, str], roles: tuple[str, ...] = ()
) -> dict:
    arcs = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {"nodes": [{"dcid": qualifier}]},
        "constraintProperties": {"nodes": [{"dcid": prop} for prop in constraints]},
        **{prop: {"nodes": [{"dcid": value}]} for prop, value in constraints.items()},
    }
    if roles:
        arcs["observationProperties"] = {"nodes": [{"dcid": role} for role in roles]}
    return arcs


# The IDS series reads at a price basis off the declared default one, so a question naming
# none leaves the price basis open and GNI PPP is a rival beside it (status rule 2(a)).
_IDS_QUALIFIER = "CurrentPricesScenario"

_NODES = {
    _GNI_IDS: {
        "label": "Gross national income (current US$)",
        "type": "StatisticalVariable",
        "arcs": _economic_activity_arcs(
            _IDS_QUALIFIER,
            {
                "economicActivityType": "GrossNationalIncome",
            },
            ("financingRecipient",),
        ),
    },
    _GNI_PPP: {
        "label": "Gross national income based on purchasing power parity",
        "type": "StatisticalVariable",
        "arcs": _economic_activity_arcs(
            "PurchasingPowerParity", {"activitySource": "GrossNationalIncome"}
        ),
    },
    _GHA: {"label": "Ghana", "type": "Country"},
    _NOR: {"label": "Norway", "type": "Country"},
}

_PPP_OBS = {f"{_GNI_PPP}|{_GHA}": [{"earliestDate": "1990", "latestDate": "2024", "obsCount": 35}]}


_COVERAGE_ROWS = [(_GNI_IDS, "f_ids", "", _GHA, _YEARS)]
_ROLE_NAMES = {_GNI_IDS: RoleNames(donor=None, recipient="financingRecipient")}


def _coverage() -> FakeCoverageService:
    return FakeCoverageService(build_coverage_index(_COVERAGE_ROWS, role_names=_ROLE_NAMES))


def _graph(query: str, order: list[str], scores: list[float]) -> FakeGraph:
    return FakeGraph(
        nodes=_NODES,
        obs=_PPP_OBS,
        detect={query: {"svs": order, "cosine_scores": scores}},
        resolve={"Ghana": _GHA, "Norway": _NOR},
    )


class _ElectFirstWithAlternate:
    """Elects the first shape on the menu and names the second as also plausible."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, bindings=[], also_plausible=[1]), None


def _resolve(query: str, order: list[str], scores: list[float], entities: list[str]):
    graph = _graph(query, order, scores)
    return asyncio.run(
        resolve_variable(
            query,
            entities=entities,
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_ElectFirstWithAlternate(),
            sdmx=FakeSdmxClient(observations={}, scope_availability=graph.scope_availability),
            base_timing={},
            coverage=_coverage().snapshot(),
        )
    )


def _spec_serving(result, stat_var: str):
    specs = [
        spec
        for spec in result.specs
        if [item.ref.ref for item in spec.resolved.members] == [stat_var]
    ]
    assert len(specs) == 1, [spec.resolved.stat_vars for spec in result.specs]
    return specs[0]


def _options_graph(
    stat_var: str,
    *,
    qualifier: str,
    constraints: dict[str, str],
    roles: tuple[str, ...] = (),
    facet_id: str,
) -> FakeGraphReader:
    """A materialise graph serving one StatVar's population and its facet at Ghana."""
    arcs = {
        stat_var: {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("EconomicActivity",),
            "measuredProperty": ("amount",),
            "statType": ("measuredValue",),
            "measurementQualifier": (qualifier,),
            "measurementDenominator": (),
            "constraintProperties": tuple(constraints),
            **{prop: (value,) for prop, value in constraints.items()},
            "observationProperties": roles,
        }
    }
    facet = MaterialiseFacet(
        facet_id=facet_id,
        provenance_id="dc/base/WorldDevelopmentIndicators",
        measurement_method=None,
        unit=None,
        obs_count=35,
        earliest_date="1990",
        latest_date="2024",
    )
    return FakeGraphReader(
        arcs=arcs,
        subjects={("amount", "measuredProperty"): SubjectPage(dcids=(stat_var,), truncated=False)},
        facets={(stat_var, _GHA): [facet]},
    )


def _ppp_options_graph() -> FakeGraphReader:
    return _options_graph(
        _GNI_PPP,
        qualifier="PurchasingPowerParity",
        constraints={"activitySource": "GrossNationalIncome"},
        facet_id="f_ppp",
    )


def _ids_options_graph() -> FakeGraphReader:
    return _options_graph(
        _GNI_IDS,
        qualifier=_IDS_QUALIFIER,
        constraints={
            "economicActivityType": "GrossNationalIncome",
        },
        roles=("financingRecipient",),
        facet_id="f_ids",
    )


def _options_status(graph: FakeGraph, materialise_graph: FakeGraphReader, request) -> Any:
    client = TestClient(
        search_app(
            graph=graph,
            resolver=make_resolver(graph),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(observations={}, scope_availability=graph.scope_availability),
            materialise_graph=materialise_graph,
        ),
        raise_server_exceptions=False,
    )
    return client.post("/api/search/options", json=request.model_dump(by_alias=True, mode="json"))


def test_subject_only_alternate_beside_a_role_bearing_election_binds_observation_about() -> None:
    """GNI PPP, offered beside the IDS series, mints with Ghana as its subject, and the
    minted identity resubmits and opens its place options."""
    query = "gross national income of Ghana"
    result = _resolve(query, [_GNI_IDS, _GNI_PPP], [0.75, 0.74], ["Ghana"])

    assert result.status == "candidates", result
    ids_spec = _spec_serving(result, _GNI_IDS)
    ppp_spec = _spec_serving(result, _GNI_PPP)
    assert ids_spec.definition.identity.entities == {"destination": [_GHA]}
    assert ppp_spec.definition.identity.entities == {"observationAbout": [_GHA]}
    assert [entity.role for entity in ppp_spec.resolved.entities] == ["observationAbout"]

    graph = _graph(query, [_GNI_IDS, _GNI_PPP], [0.75, 0.74])
    replay = offline_resolve(
        ResolveRequest(
            input=DatasetResubmitInput(entries=[_identity_entry_from_spec(ppp_spec, query)])
        ),
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, scope_availability=graph.scope_availability),
    )
    assert replay.parts[0].status == "definite", replay.parts[0]
    assert replay.parts[0].spec.spec_id == ppp_spec.spec_id

    request = OptionsRequest(
        identity=ppp_spec.definition.identity, **{"for": OptionsFor(role="observationAbout")}
    )
    response = _options_status(graph, _ppp_options_graph(), request)
    assert response.status_code == 200, response.json()


def test_role_bearing_alternate_beside_a_subject_only_election_binds_its_required_role(
    monkeypatch,
) -> None:
    """The IDS series, offered beside GNI PPP, binds Ghana on its one required role and
    serves Ghana's own series, and the minted identity resubmits to the same spec and opens
    its recipient options."""
    # The resubmit door reads the process coverage service.
    patch_coverage(monkeypatch, _COVERAGE_ROWS, role_names=_ROLE_NAMES)
    query = "gross national income of Ghana"
    result = _resolve(query, [_GNI_PPP, _GNI_IDS], [0.75, 0.74], ["Ghana"])

    assert result.status == "candidates", result
    ppp_spec = _spec_serving(result, _GNI_PPP)
    ids_spec = _spec_serving(result, _GNI_IDS)
    assert ppp_spec.definition.identity.entities == {"observationAbout": [_GHA]}
    assert ids_spec.definition.identity.entities == {"destination": [_GHA]}
    assert ids_spec.resolved.scope is None

    graph = _graph(query, [_GNI_PPP, _GNI_IDS], [0.75, 0.74])
    replay = offline_resolve(
        ResolveRequest(
            input=DatasetResubmitInput(entries=[_identity_entry_from_spec(ids_spec, query)])
        ),
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, scope_availability=graph.scope_availability),
    )
    assert replay.parts[0].status == "definite", replay.parts[0]
    assert replay.parts[0].spec.spec_id == ids_spec.spec_id

    request = OptionsRequest(
        identity=ids_spec.definition.identity, **{"for": OptionsFor(role="destination")}
    )
    response = _options_status(graph, _ids_options_graph(), request)
    assert response.status_code == 200, response.json()
    assert [item["value"]["ref"]["ref"] for item in response.json()["options"]["items"]] == [_GHA]


class _ElectFirstOnly:
    """Elects the first shape on the menu and names no alternate."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, bindings=[], also_plausible=[]), None


_GNI_IDS_BY_CREDITOR = "Amount_EconomicActivity_GrossNationalIncome_Nominal_ByCreditor"
_USA = "country/USA"
_MIXED_QUERY = "gross national income of Ghana from Norway"
_MIXED_ENTITIES = ["Ghana", "Norway"]
_MIXED_ORDER, _MIXED_SCORES = [_GNI_IDS, _GNI_IDS_BY_CREDITOR, _GNI_PPP], [0.85, 0.85, 0.84]


_MIXED_NODES = {
    **_NODES,
    _GNI_IDS_BY_CREDITOR: {
        "label": "Gross national income (current US$), by creditor",
        "type": "StatisticalVariable",
        "arcs": _economic_activity_arcs(
            _IDS_QUALIFIER,
            {
                "economicActivityType": "GrossNationalIncome",
            },
            ("financingProvider", "financingRecipient"),
        ),
    },
    _USA: {"label": "United States", "type": "Country"},
}


def _mixed_graph() -> FakeGraph:
    """GNI PPP with data at Ghana and Norway, beside an IDS shape grouping a recipient-only
    member with a provider-and-recipient twin."""
    return FakeGraph(
        nodes=_MIXED_NODES,
        obs={
            f"{_GNI_PPP}|{_GHA}": [
                {
                    "earliestDate": "2000",
                    "latestDate": "2020",
                    "obsCount": 21,
                    "provenanceId": "dc/base/WorldDevelopmentIndicators",
                }
            ],
            f"{_GNI_PPP}|{_NOR}": [
                {
                    "earliestDate": "1990",
                    "latestDate": "2024",
                    "obsCount": 35,
                    "provenanceId": "dc/base/WorldDevelopmentIndicators",
                }
            ],
        },
        detect={_MIXED_QUERY: {"svs": _MIXED_ORDER, "cosine_scores": _MIXED_SCORES}},
        resolve={"Ghana": _GHA, "Norway": _NOR},
    )


def _resolve_mixed():
    """Resolve the mixed query with the IDS shape elected and data only at the United States."""
    coverage = FakeCoverageService(
        build_coverage_index(
            [
                (_GNI_IDS, "f_ids", "", _USA, _YEARS),
                (_GNI_IDS_BY_CREDITOR, "f_ids", _USA, _USA, _YEARS),
            ],
            role_names={
                _GNI_IDS: RoleNames(donor=None, recipient="financingRecipient"),
                _GNI_IDS_BY_CREDITOR: RoleNames(
                    donor="financingProvider", recipient="financingRecipient"
                ),
            },
        )
    )
    graph = _mixed_graph()
    return asyncio.run(
        resolve_variable(
            _MIXED_QUERY,
            entities=_MIXED_ENTITIES,
            date_request=None,
            detect_query=_MIXED_QUERY,
            role_query=_MIXED_QUERY,
            entity_roles=door_roles_for_test_query(_MIXED_QUERY, _MIXED_ENTITIES),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_ElectFirstOnly(),
            sdmx=FakeSdmxClient(observations={}, scope_availability=graph.scope_availability),
            base_timing={},
            coverage=coverage.snapshot(),
        )
    )


def _resolve_electing_first(
    query: str, entities: list[str], *, graph: FakeGraph, coverage_rows: list, role_names: dict
):
    """Resolve `query` with the first shape elected, no alternate named and door roles read
    from the query's prepositions."""
    coverage = FakeCoverageService(build_coverage_index(coverage_rows, role_names=role_names))
    return asyncio.run(
        resolve_variable(
            query,
            entities=entities,
            date_request=None,
            detect_query=query,
            role_query=query,
            entity_roles=door_roles_for_test_query(query, entities),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_ElectFirstOnly(),
            sdmx=FakeSdmxClient(observations={}, scope_availability=graph.scope_availability),
            base_timing={},
            coverage=coverage.snapshot(),
        )
    )


_KEN = "country/KEN"
_GNI_REAL = "Amount_EconomicActivity_GrossNationalIncome_RealValue"
_GNI_REAL_BY_CREDITOR = "Amount_EconomicActivity_GrossNationalIncome_RealValue_ByCreditor"
_GNI_ATLAS = "Amount_EconomicActivity_GrossNationalIncome_AtlasMethod"
_GHA_OBS = [{"earliestDate": "2000", "latestDate": "2020", "obsCount": 21}]
_MIXED_ROLE_NAMES = {
    _GNI_IDS: RoleNames(donor=None, recipient="financingRecipient"),
    _GNI_IDS_BY_CREDITOR: RoleNames(donor="financingProvider", recipient="financingRecipient"),
    _GNI_REAL: RoleNames(donor=None, recipient="financingRecipient"),
    _GNI_REAL_BY_CREDITOR: RoleNames(donor="financingProvider", recipient="financingRecipient"),
}


_TWO_RECIPIENTS_QUERY = "gross national income to Ghana and Kenya"


_NORDIA = "nordia"
_NORDIA_QUERY = "gross national income of Ghana from Nordia"
_NORDIA_ENTITIES = ["Ghana", "Nordia"]
# Nordia's rows as the provider of Ghana's income, which give the IDS shape data.
_NORDIA_ROWS = [
    (_GNI_IDS_BY_CREDITOR, "f_ids", _NORDIA, _GHA, _YEARS),
    (_GNI_IDS, "f_ids", "", _GHA, _YEARS),
]
# Rows that leave the IDS shape without data at Ghana.
_ELSEWHERE_ROWS = [
    (_GNI_IDS, "f_ids", "", _USA, _YEARS),
    (_GNI_IDS_BY_CREDITOR, "f_ids", _USA, _USA, _YEARS),
]


_NORDIA_MEMBERS = ["country/DNK", "country/NOR", "country/SWE"]


@pytest.fixture(autouse=True)
def _regions(declare_regions):
    """Search reads Nordia's members from the membership policy."""
    declare_regions({_NORDIA: _NORDIA_MEMBERS})


def _nordia_graph(order: list[str], scores: list[float], query: str = _NORDIA_QUERY) -> FakeGraph:
    """Nordia, a continent of three countries the provider role admits, beside Ghana. The
    IDS shape binds Ghana as recipient and Nordia as provider, as Nordia's node where it
    holds rows as Ghana's provider (two places) and otherwise as its three members (four
    places). GNI PPP
    holds no observation at Nordia's node, so it binds Ghana and Nordia's three members as
    its subjects, four places."""
    members = _NORDIA_MEMBERS
    return FakeGraph(
        nodes={
            **_MIXED_NODES,
            _NORDIA: {"label": "Nordia", "type": "Continent", "arcs": {}},
            **{dcid: {"label": dcid, "type": "Country"} for dcid in members},
            "financingProvider": {
                "arcs": {"rangeIncludes": {"nodes": [{"dcid": "Country"}, {"dcid": "Continent"}]}}
            },
        },
        obs={f"{_GNI_PPP}|{_GHA}": _GHA_OBS},
        detect={query: {"svs": order, "cosine_scores": scores}},
        resolve={"Ghana": _GHA, "Norway": _NOR},
        places={"Nordia": {"dcid": _NORDIA, "type_of": "Continent"}},
        children={_NORDIA: {"Country": members}},
    )


def _resolve_nordia(
    order: list[str],
    scores: list[float],
    *,
    coverage_rows: list,
    llm,
    operations: Sequence[Operation] = (),
    query: str = _NORDIA_QUERY,
    entities: Sequence[str] = tuple(_NORDIA_ENTITIES),
):
    graph = _nordia_graph(order, scores, query)
    coverage = FakeCoverageService(
        build_coverage_index(coverage_rows, role_names=_MIXED_ROLE_NAMES)
    )
    return asyncio.run(
        resolve_variable(
            query,
            entities=list(entities),
            date_request=None,
            detect_query=query,
            role_query=query,
            entity_roles=door_roles_for_test_query(query, list(entities)),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=llm,
            sdmx=FakeSdmxClient(observations={}, scope_availability=graph.scope_availability),
            base_timing={},
            coverage=coverage.snapshot(),
            operations=operations,
        )
    )


def _served_members(result) -> list[list[str]]:
    return [[item.ref.ref for item in spec.resolved.members] for spec in result.specs]


def _places_bound(result) -> list[int]:
    """The dcids each served spec binds, distinct per role and summed over its roles."""
    return [
        sum(len(set(dcids)) for dcids in spec.definition.identity.entities.values())
        for spec in result.specs
    ]


# A rank over the recipients: Ghana, the one recipient of the IDS shape, is no set, and
# Nordia is a subject of GNI PPP, so the rank asks for Nordia's members there alone.
_RANK_RECIPIENTS = (Operation(verb="rank", phrase="highest", over="destination"),)


@pytest.mark.parametrize("operations", [(), _RANK_RECIPIENTS], ids=["read-decides", "rank-decides"])
@pytest.mark.parametrize(
    ("bound", "status", "served"),
    [
        pytest.param(None, "candidates", [[_GNI_IDS_BY_CREDITOR], [_GNI_PPP]], id="default-bound"),
        pytest.param(3, "definite", [[_GNI_IDS_BY_CREDITOR]], id="bound-of-three"),
    ],
)
def test_alternate_over_the_place_bound_is_dropped_and_the_elected_shape_served(
    search_settings, bound, status, served, operations
) -> None:
    """The elected IDS shape binds two places and its alternate GNI PPP four. Under the
    default bound both are offered. Under a bound of three the alternate is dropped whole,
    the elected shape is served, and no spec binds more places than the bound. The
    alternate binds Nordia's members because its own read finds no data at Nordia's node,
    or, before any read, because a rank runs over its subjects."""
    if bound is not None:
        search_settings(max_child_places=bound)
    result = _resolve_nordia(
        [_GNI_IDS, _GNI_IDS_BY_CREDITOR, _GNI_PPP],
        [0.85, 0.85, 0.84],
        coverage_rows=_NORDIA_ROWS,
        llm=_ElectFirstWithAlternate(),
        operations=operations,
    )

    assert result.status == status, result
    assert _served_members(result) == served
    assert _places_bound(result) == [2, 4][: len(served)]


_GNI_TRADE = "Amount_TradeActivity_GrossNationalIncome"
_GNI_AID = "Amount_AidActivity_GrossNationalIncome"
_GNI_REMITTED_BY_CREDITOR = "Amount_RemittanceActivity_GrossNationalIncome_ByCreditor"


def _population_arcs(population: str, roles: tuple[str, ...] = ()) -> dict:
    """The current-price GNI arcs over another population, a semantically distinct reading."""
    return {
        **_economic_activity_arcs(
            "Nominal", {"economicActivityType": "GrossNationalIncome"}, roles
        ),
        "populationType": {"nodes": [{"dcid": population}]},
    }


def _admissions(monkeypatch) -> list[Any]:
    """Record each candidate admission (`_AlternatesAdmission`) the request makes."""
    real = candidate_admission.admit_alternates
    seen: list[Any] = []

    async def _admit(state: Any) -> Any:
        admission = await real(state)
        seen.append(admission)
        return admission

    monkeypatch.setattr(candidate_admission, "admit_alternates", _admit)
    return seen


def test_an_alternate_over_the_place_bound_is_not_reported_read_and_empty(
    monkeypatch, search_settings
) -> None:
    """The alternate GNI PPP, read at four places over a bound of three, is dropped unread:
    admission lists it neither as admitted nor as rejected, so no alternate counts as
    evaluated, and the elected shape is served down the single-shape path with its
    `options` step."""
    search_settings(max_child_places=3)
    admissions = _admissions(monkeypatch)
    result = _resolve_nordia(
        [_GNI_IDS, _GNI_IDS_BY_CREDITOR, _GNI_PPP],
        [0.85, 0.85, 0.84],
        coverage_rows=_NORDIA_ROWS,
        llm=_ElectFirstWithAlternate(),
    )

    (admission,) = admissions
    assert (admission.admitted, admission.rejected, admission.evaluated) == ([], [], False)
    assert (result.status, result.no_data_reason) == ("definite", None)
    assert "options" in result.timing_by_step


@pytest.mark.parametrize(
    ("bound", "status", "reason", "served"),
    [
        pytest.param(None, "no_data", "no_observations", [], id="default-bound"),
        pytest.param(3, "no_data", "entity_set_too_large", [], id="bound-of-three"),
    ],
)
def test_an_elected_shape_with_no_data_at_a_node_binds_the_members_before_reelection(
    search_settings, bound, status, reason, served
) -> None:
    """The elected IDS shape has no data at Ghana from Nordia's node, so it binds Nordia's
    three members beside Ghana. Under the default bound it has no data there either, and
    GNI PPP, a subject-only reading, answers for no role-bearing shape (status rule 1), so
    the part refuses. Under a bound of three the elected shape's four places are over the
    bound."""
    if bound is not None:
        search_settings(max_child_places=bound)
    result = _resolve_nordia(
        [_GNI_IDS, _GNI_IDS_BY_CREDITOR, _GNI_PPP],
        [0.85, 0.85, 0.84],
        coverage_rows=_ELSEWHERE_ROWS,
        llm=_ElectFirstOnly(),
    )

    assert result.status == status, result
    assert result.no_data_reason == reason
    assert _served_members(result) == served


def test_reelection_sibling_over_the_place_bound_is_dropped(search_settings) -> None:
    """Elected GNI PPP holds no data at Norway or at Nordia's members, three subjects in
    all since Norway is one of them. A rank over the providers asks for Nordia's members
    on its sibling the IDS shape too, which binds Norway as recipient and those three as
    providers: four places, over a bound of three before any read. The sibling is dropped
    and the elected shape keeps its own no-data answer."""
    search_settings(max_child_places=3)
    query = "gross national income of Norway from Nordia"

    result = _resolve_nordia(
        [_GNI_PPP, _GNI_IDS, _GNI_IDS_BY_CREDITOR],
        [0.85, 0.84, 0.84],
        coverage_rows=_ELSEWHERE_ROWS,
        llm=_ElectFirstOnly(),
        operations=(Operation(verb="rank", phrase="largest", over="origin"),),
        query=query,
        entities=("Norway", "Nordia"),
    )

    assert result.status == "no_data", result
    assert result.no_data_reason == "no_observations"
    assert result.specs == ()


def test_elected_shape_over_the_place_bound_is_not_reelected(search_settings) -> None:
    """Elected GNI PPP is read at four places, over a bound of three, and answers
    `entity_set_too_large`. The IDS shape holds data within the bound and is not elected
    in its place."""
    search_settings(max_child_places=3)
    result = _resolve_nordia(
        [_GNI_PPP, _GNI_IDS, _GNI_IDS_BY_CREDITOR],
        [0.85, 0.84, 0.84],
        coverage_rows=_NORDIA_ROWS,
        llm=_ElectFirstWithAlternate(),
    )

    assert result.status == "no_data", result
    assert result.no_data_reason == "entity_set_too_large"
    assert result.specs == ()


def test_placeless_role_bearing_shape_takes_no_default_scope() -> None:
    """With no place named, the elected IDS series is role-bearing, so its probe scope stays
    open: the answer carries no place_scope_defaulted warning and Earth's countries are
    never enumerated for a default scope."""
    query = "gross national income"
    enumerated: list[tuple[list[str], str]] = []

    class _Graph(FakeGraph):
        def child_places_batch(self, dcids, child_type, *args, **kwargs):
            enumerated.append((list(dcids), child_type))
            return super().child_places_batch(dcids, child_type, *args, **kwargs)

    graph = _Graph(
        nodes=_NODES,
        obs={},
        detect={query: {"svs": [_GNI_IDS], "cosine_scores": [0.85]}},
        resolve={},
        children={"Earth": {"Country": [_GHA, _NOR]}},
    )
    result = _resolve_electing_first(
        query, [], graph=graph, coverage_rows=_COVERAGE_ROWS, role_names=_ROLE_NAMES
    )

    assert result.status == "definite", result
    assert [item.ref.ref for item in result.specs[0].resolved.members] == [_GNI_IDS]
    assert "place_scope_defaulted" not in [w.code for w in result.warnings]
    assert (["Earth"], "Country") not in enumerated


def test_alternate_whose_role_cannot_take_the_place_is_dropped() -> None:
    """Norway named as the origin binds as GNI PPP's subject, but the IDS series declares
    only a recipient, so it declares no role on the axis the query directs Norway to and is
    no candidate (`lacks_query_axis`): it never reaches bind, so no binding of it fails."""
    query = "gross national income from Norway"
    graph = _graph(query, [_GNI_PPP, _GNI_IDS], [0.75, 0.74])
    graph_obs = {
        f"{_GNI_PPP}|{_NOR}": [{"earliestDate": "1990", "latestDate": "2024", "obsCount": 35}]
    }
    graph = FakeGraph(
        nodes=_NODES,
        obs=graph_obs,
        detect={query: {"svs": [_GNI_PPP, _GNI_IDS], "cosine_scores": [0.75, 0.74]}},
        resolve={"Norway": _NOR},
    )

    binder = MenuRecordingBinder()
    result = asyncio.run(
        resolve_variable(
            query,
            entities=["Norway"],
            date_request=None,
            detect_query=query,
            role_query=query,
            entity_roles=door_roles_for_test_query(query, ["Norway"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_ElectFirstWithAlternate(),
            binder=binder,
            sdmx=FakeSdmxClient(observations={}, scope_availability=graph.scope_availability),
            base_timing={},
            coverage=_coverage().snapshot(),
        )
    )

    assert result.status == "definite", result
    assert [item.ref.ref for item in result.specs[0].resolved.members] == [_GNI_PPP]
    assert result.specs[0].definition.identity.entities == {"observationAbout": [_NOR]}
    assert binder.menus == [[(_GNI_PPP,)]]


def _shape_for(arcs: dict) -> ShapeDraft:
    nodes = {"Test_PlaceRolesShape": {"label": "Test shape", "arcs": arcs}}
    (shape,) = derive_shapes(
        confirmed_svs=list(nodes), graph=FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
    )
    return shape


def _state(*, names_and_roles: dict[str, list[DoorRole]], places: dict[str, str]) -> ResolveState:
    def _none(*_: object) -> Any:
        return None

    return ResolveState(
        variable="flow",
        entities=list(places),
        entity_roles=names_and_roles,
        place_sets=place_sets(places),
        role_bearing_outranks=True,
        date_request=None,
        detect_query="flow",
        role_query="flow",
        pac=True,
        graph=_none(),
        resolver=_none(),
        admission=_none(),
        llm=_none(),
        sdmx=_none(),
        deadline=None,
        pre_resolved=None,
        variable_index=0,
        emit=None,
        include_option_coverage=False,
        stream_slot_options=False,
        include_observations=False,
        stream_observations=False,
        unit_text=None,
        materialise_graph=None,
        materialise_sdmx=None,
        timing={},
        warnings=[],
        coverage=empty_coverage_snapshot(),
    )


_CRS = "OECDCreditorReportingSystem"
_STATED = frozenset({"bilateral"})
_NOTHING: frozenset[str] = frozenset()


def _door_role(axis: str, name: str = "Kenya") -> DoorRole:
    return door_roles_for_test_query(f"aid {'from' if axis == 'origin' else 'to'} {name}", [name])[
        name
    ][0]


def _state_naming(variable: str, party: str | None, axis: str = "destination") -> ResolveState:
    """A state whose `variable` names one party on `axis`: a country or an organisation
    that resolves to its kind, a region or a group of countries, a group of countries whose
    own node is an organisation the recipient role admits, as the EU's, or no party. As on
    the dev graph, the recipient role admits no continent, so a region reads at its members
    or its total."""
    state = _state(names_and_roles={}, places={})
    state.variable = variable
    if party is None:
        return state
    name = "Kenya" if party in ("country", "organisation") else "Africa"
    if party in ("country", "organisation"):
        state.place_sets = place_sets({name: _KEN})
        graph_type = "Country" if party == "country" else "Organization"
        entity = Entity(dcid=_KEN, name=name, kind=party, graph_type=graph_type)
        state.rcl = Recall(candidates=[], resolutions={name: Resolved(entity=entity)})
    elif party == "region":
        state.place_sets = place_sets(regions={name: ("africa", [_KEN, _GHA])})
        state.node_admissions[("financingRecipient", "Continent")] = False
    elif party == "group-at-organisation":
        name = "the EU"
        state.place_sets = {
            name: PlaceSet(
                name=name,
                kind="group",
                node=PlaceNode(dcid="EuropeanUnionInstitutions", graph_type="Organization"),
                members=("country/DEU", "country/FRA"),
                member_type="Country",
            )
        }
        state.node_admissions[("financingRecipient", "Organization")] = True
    else:
        state.place_sets = place_sets(groups={name: [_KEN, _GHA]})
    state.entities = [name]
    state.entity_roles = {name: [_door_role(axis, name)]}
    return state


# The aid type each DAC2 table's series hold, which reads "ODA" on DAC2a and "OOF" on DAC2b.
_AID_TYPES = {
    "OECDDAC2A": {"dac2aAidType": "DAC2aAidType206"},
    "OECDDAC2B": {"dac2bAidType": "DAC2bAidType972"},
}


def _flow_shape(provenance: str, measured_property: str = "amount") -> ShapeDraft:
    """A financial flow shape declaring both flow roles, answered from `provenance`, with
    the aid type slot of a DAC2 table."""
    roles = ("financingProvider", "financingRecipient")
    return dataclasses.replace(
        _shape_for(_economic_activity_arcs("Nominal", _AID_TYPES.get(provenance, {}), roles)),
        meas_prop_dcid=measured_property,
        datasets=(f"provenance/{provenance}",),
    )


@pytest.mark.parametrize(
    ("provenance", "measured_property", "variable", "party", "axis", "stated"),
    [
        pytest.param(_CRS, "amount", "bilateral ODA", None, "destination", _STATED, id="crs"),
        pytest.param(
            _CRS, "amount", "bilateral ODA", "organisation", "destination", _STATED, id="crs-org"
        ),
        pytest.param(
            "OECDDAC2A", "amount", "bilateral ODA", "country", "destination", _STATED, id="dac2a"
        ),
        pytest.param(
            "OECDDAC2B", "amount", "bilateral ODA", "country", "destination", _STATED, id="dac2b"
        ),
        pytest.param(
            "OECDDAC2A", "amount", "bilateral ODA", "region", "destination", _STATED, id="region"
        ),
        pytest.param(
            "OECDDAC2A", "amount", "bilateral ODA", "group", "destination", _STATED, id="group"
        ),
        pytest.param(
            "OECDDAC2A",
            "amount",
            "bilateral ODA",
            "organisation",
            "destination",
            _NOTHING,
            id="dac2a-organisation",
        ),
        pytest.param(
            "OECDDAC2A",
            "amount",
            "bilateral ODA",
            "group-at-organisation",
            "destination",
            _NOTHING,
            id="dac2a-group-at-an-organisation-node",
        ),
        pytest.param(
            "OECDDAC2A", "amount", "bilateral ODA", None, "destination", _NOTHING, id="dac2a-open"
        ),
        pytest.param(
            "OECDDAC2A",
            "amount",
            "bilateral ODA",
            "country",
            "origin",
            _NOTHING,
            id="dac2a-recipient-open",
        ),
        pytest.param(
            "OECDDAC2A",
            "attributedAmount",
            "bilateral ODA",
            "country",
            "destination",
            _NOTHING,
            id="dac2a-attributed",
        ),
        pytest.param(
            "ONEImputedMultilateralSectors",
            "attributedAmount",
            "bilateral ODA",
            "country",
            "destination",
            _NOTHING,
            id="imputed",
        ),
        pytest.param(
            "OECDDAC1", "amount", "bilateral ODA", "country", "destination", _NOTHING, id="dac1"
        ),
        pytest.param(
            _CRS, "amount", "non-bilateral ODA", None, "destination", _NOTHING, id="crs-hyphen"
        ),
        pytest.param(
            _CRS,
            "amount",
            "ODA excluding bilateral flows",
            None,
            "destination",
            _NOTHING,
            id="crs-excluded",
        ),
        pytest.param(
            _CRS, "amount", "net bilateral ODA", None, "destination", _STATED, id="crs-modifier"
        ),
        pytest.param(
            _CRS,
            "amount",
            "ODA net of bilateral flows",
            None,
            "destination",
            _NOTHING,
            id="crs-past-a-function-word",
        ),
        pytest.param(
            _CRS,
            "amount",
            "ODA, bilateral excluded",
            None,
            "destination",
            _NOTHING,
            id="crs-past-a-separator",
        ),
    ],
)
def test_a_dataset_states_a_word_where_the_parts_party_is_of_the_kind_it_names(
    provenance, measured_property, variable, party, axis, stated
) -> None:
    """The CRS states "bilateral" of every `amount` row. DAC2a and DAC2b state it of `amount`
    rows whose recipient is a country, or a region or group of countries, and not where the
    part names an organisation, a group that can settle at an organisation node, or leaves
    the recipient open. No table states it of an
    attributed amount. The part states the word where it modifies the measure: a negated
    or excluded occurrence, and one past a function word or a separator, is no statement.
    A resubmit gives no state and reads none."""
    state = _state_naming(variable, party, axis)
    shape = _flow_shape(provenance, measured_property)

    assert _dataset_stated_words(shape, get_index(), state) == stated
    assert _dataset_stated_words(shape, get_index(), None) == _NOTHING


@pytest.mark.parametrize(
    ("variable", "party", "provenance", "words", "query_text"),
    [
        pytest.param(
            "bilateral ODA", None, _CRS, _STATED, "not bilateral ODA to Kenya", id="stated"
        ),
        pytest.param(
            "multilateral ODA",
            "country",
            "OECDDAC2A",
            frozenset({"multilateral"}),
            "not multilateral ODA to Kenya",
            id="party",
        ),
    ],
)
def test_a_negation_the_variable_left_out_is_read_from_the_query(
    variable, party, provenance, words, query_text
) -> None:
    """The extraction can leave a negation out of the variable, and the verbatim query
    still holds it, for a word a dataset states and one that names a party alike."""
    state = _with_multilaterals(_state_naming(variable, party))
    shape = _flow_shape(provenance)
    assert _dataset_stated_words(shape, get_index(), state) == words

    state.query_text = query_text

    assert _dataset_stated_words(shape, get_index(), state) == _NOTHING
    assert held_party_words(state, shape) == ()


_MULTILATERALS = "DACMultilaterals"
_FRA = "country/FRA"


def _with_multilaterals(state: ResolveState) -> ResolveState:
    """`state` holding the place set intake reads for the party "multilateral" names on
    DAC2a and DAC2b (`ResolveState.party_sets`)."""
    state.party_sets = {
        _MULTILATERALS: PlaceSet(
            name="multilaterals (DAC)",
            kind="place",
            node=PlaceNode(dcid=_MULTILATERALS, graph_type="DACGroup"),
        )
    }
    return state


def _naming_france_as_provider(state: ResolveState) -> ResolveState:
    """`state` with France named as the provider beside its own places."""
    state.place_sets = {**state.place_sets, **place_sets({"France": _FRA})}
    state.entities = ["France", *state.entities]
    state.entity_roles = {**state.entity_roles, "France": [_door_role("origin", "France")]}
    return state


@pytest.mark.parametrize(
    ("datasets", "measured_property", "variable", "recipient", "change", "held"),
    [
        pytest.param(
            ("OECDDAC2A",), "amount", "multilateral ODA", "country", None, True, id="dac2a"
        ),
        pytest.param(
            ("OECDDAC2B",), "amount", "multilateral OOF", "country", None, True, id="dac2b"
        ),
        pytest.param(
            ("OECDDAC2A", "OECDDAC2B"),
            "amount",
            "multilateral ODA",
            "country",
            None,
            True,
            id="every-dataset-declares-it",
        ),
        pytest.param(
            ("OECDDAC2A", _CRS),
            "amount",
            "multilateral ODA",
            "country",
            None,
            False,
            id="one-dataset-declares-nothing",
        ),
        pytest.param((_CRS,), "amount", "multilateral ODA", "country", None, False, id="crs"),
        pytest.param(
            ("OECDDAC1",), "amount", "multilateral ODA", "country", None, False, id="dac1"
        ),
        pytest.param(
            ("OECDDAC2A",),
            "attributedAmount",
            "imputed multilateral ODA",
            "country",
            None,
            False,
            id="dac2a-attributed",
        ),
        pytest.param(
            ("OECDDAC2A",), "amount", "net multilateral ODA", "country", None, True, id="net"
        ),
        pytest.param(
            ("OECDDAC2A",), "amount", "multilateral ODA", "region", None, True, id="region"
        ),
        pytest.param(
            ("OECDDAC2A",),
            "amount",
            "multilateral ODA",
            "organisation",
            None,
            False,
            id="organisation-recipient",
        ),
        pytest.param(
            ("OECDDAC2A",), "amount", "multilateral ODA", None, None, False, id="recipient-open"
        ),
        pytest.param(
            ("OECDDAC2A",),
            "amount",
            "multilateral ODA",
            "country",
            "named-provider",
            False,
            id="named-provider",
        ),
        pytest.param(
            ("OECDDAC2A",),
            "amount",
            "multilateral ODA",
            "country",
            "by-donor",
            False,
            id="operation-over-the-provider",
        ),
        pytest.param(
            ("OECDDAC2A",),
            "amount",
            "multilateral ODA",
            "country",
            "every-provider",
            False,
            id="provider-axis-open",
        ),
        pytest.param(
            ("OECDDAC2A",),
            "amount",
            "multilateral ODA",
            "country",
            "recipient-only-shape",
            False,
            id="shape-declares-no-provider",
        ),
        pytest.param(
            ("OECDDAC2A",),
            "amount",
            "non-multilateral ODA",
            "country",
            None,
            False,
            id="hyphen",
        ),
        pytest.param(
            ("OECDDAC2A",),
            "amount",
            "ODA excluding multilateral flows",
            "country",
            None,
            False,
            id="excluded",
        ),
        pytest.param(
            ("OECDDAC2A",),
            "amount",
            "multilateral ODA",
            "country",
            "no-party-set",
            False,
            id="no-party-set",
        ),
    ],
)
def test_a_word_names_a_party_where_every_dataset_declares_it_and_its_axis_is_unnamed(
    datasets, measured_property, variable, recipient, change, held
) -> None:
    """DAC2a and DAC2b name `DACMultilaterals` the provider of an `amount` row to a country
    where the part states "multilateral". The shape binds the party as its provider, beside
    the recipient the part names, when every dataset it answers from declares the word, the
    recipient is a country or a region of countries, and the provider axis is unnamed: no
    place, no operation across it, and no request for its every member, on a shape that
    declares it. The word then reads as stated. A part intake read no party set for binds
    none."""
    state = _with_multilaterals(_state_naming(variable, recipient))
    if change == "named-provider":
        _naming_france_as_provider(state)
    elif change == "by-donor":
        state.operations = (Operation(verb="rank", phrase="by donor", over="origin"),)
    elif change == "every-provider":
        state.open_axes = frozenset({"origin"})
    elif change == "no-party-set":
        state.party_sets = {}
    shape = dataclasses.replace(
        _flow_shape(datasets[0], measured_property),
        datasets=tuple(f"provenance/{dataset}" for dataset in datasets),
    )
    if change == "recipient-only-shape":
        shape = dataclasses.replace(shape, declared_roles=("financingRecipient",))

    assignment = bind_place_roles(state, shape)

    assert not isinstance(assignment, str)
    assert [named.word for named in held_party_words(state, shape)] == (
        ["multilateral"] if held else []
    )
    assert ("multilateral" in _dataset_stated_words(shape, get_index(), state)) == held
    providers = [place_set.name for place_set in assignment.sets.get("origin", ())]
    if held:
        assert providers == ["multilaterals (DAC)"]
    else:
        assert "multilaterals (DAC)" not in providers


@pytest.mark.parametrize(
    ("variable", "change", "words"),
    [
        pytest.param("multilateral ODA", None, ["multilateral"], id="recipient-named"),
        pytest.param("multilateral ODA", "named-provider", [], id="provider-named"),
        pytest.param("multilateral ODA", "by-donor", [], id="operation-over-the-provider"),
        pytest.param("multilateral ODA", "every-provider", [], id="provider-axis-open"),
        pytest.param("multilateral ODA", "negated-in-the-query", [], id="negated"),
        pytest.param("ODA from multilaterals", None, [], id="another-word"),
        pytest.param("multi-bi aid", None, [], id="another-token"),
        pytest.param(
            "contributions to multilateral organisations", None, [], id="after-a-preposition"
        ),
        pytest.param("multilateral ODA", "no-recipient", [], id="no-recipient-for-its-where"),
        pytest.param("multilateral ODA", "untyped", ["multilateral"], id="party-left-the-graph"),
        pytest.param("multilateral ODA", "party-words-off", [], id="read-without-party-words"),
    ],
)
def test_intake_reads_the_party_of_a_word_stated_on_an_unnamed_axis(
    variable, change, words, caplog
):
    """Recall searches without each word the part states on an axis the request leaves
    unnamed, by the door's directions alone, and intake reads its party's node type in one
    graph read, naming the set by the index's label. A part stating none reads nothing:
    one naming no recipient for the word's `where`, and "contributions to multilateral
    organisations", where the word follows a preposition, and one the pipeline reads
    without party words (`ResolveState.party_words_off`). A declared party the graph gives
    no type gets no set, and the read is logged with its dcid and noted as degraded."""
    state = _state_naming(variable, None if change == "no-recipient" else "country")
    if change == "named-provider":
        _naming_france_as_provider(state)
    elif change == "by-donor":
        state.operations = (Operation(verb="rank", phrase="by donor", over="origin"),)
    elif change == "every-provider":
        state.open_axes = frozenset({"origin"})
    elif change == "negated-in-the-query":
        state.query_text = "not multilateral ODA to Kenya"
    elif change == "party-words-off":
        state.party_words_off = True
    state.place_sets = {}
    typed = change != "untyped"
    # A part stating no word has no graph to read.
    state.graph = (
        FakeGraph(nodes={_MULTILATERALS: {"type": "DACGroup"}} if typed else {}) if words else None
    )

    assert [named.word for named in stated_party_words(state)] == words
    with recording_degraded_reads() as lost, caplog.at_level(logging.WARNING):
        sets = asyncio.run(_party_word_sets(state))
    assert {dcid: (s.name, s.node) for dcid, s in sets.items()} == (
        {_MULTILATERALS: ("multilaterals (DAC)", PlaceNode(_MULTILATERALS, "DACGroup"))}
        if words and typed
        else {}
    )
    assert lost.degraded == (not typed)
    assert [_MULTILATERALS in record.getMessage() for record in caplog.records] == (
        [] if typed else [True]
    )


_IDS = "WorldBankInternationalDebtStatistics"
_IMPUTED = "ONEImputedMultilateralSectors"


def _debt_shape(creditor: str | None) -> ShapeDraft:
    """A by-creditor debt series of a table the index does not declare, whose one member
    holds the creditor type `creditor`, a value the graph labels and the code index does
    not know."""
    roles = ("financingProvider", "financingRecipient")
    constraints = {} if creditor is None else {"creditorType": "TestCreditorType"}
    return dataclasses.replace(
        _shape_for(_economic_activity_arcs("Nominal", constraints, roles)),
        shape_id=_IDS,
        datasets=None,
        value_labels={} if creditor is None else {"TestCreditorType": creditor},
    )


_CRS_FLOW = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSPurposeCode11110_Nominal"
)
_DAC2A_ODA = "Amount_FinancialFlow_Gross_Disbursement_DAC2aAidType206_Nominal"


@pytest.mark.parametrize(
    ("variable", "creditor", "readings", "kept"),
    [
        pytest.param(
            "multilateral ODA",
            None,
            {_CRS: ("oda",), "OECDDAC2A": ("oda",)},
            ["OECDDAC2A"],
            id="the-binding-shape-reads-all-the-others-read",
        ),
        pytest.param(
            "multilateral aid for education",
            None,
            {_CRS: ("education",)},
            None,
            id="another-shape-reads-a-sector",
        ),
        pytest.param(
            "multilateral development bank ODA",
            None,
            {_CRS: ("oda",), "OECDDAC2A": ("oda",)},
            None,
            id="no-shape-binds-beside-an-unread-word",
        ),
        pytest.param(
            "multilateral funding",
            "Multilateral creditors",
            {},
            None,
            id="another-shape-holds-a-value-the-graph-labels-by-the-word",
        ),
        pytest.param(
            "multilateral funding",
            "Bilateral creditors",
            {},
            ["OECDDAC2A"],
            id="another-value-does-not",
        ),
        pytest.param(
            "multilateral funding",
            None,
            {_IDS: ("multilateral funding",)},
            None,
            id="another-shape-codes-the-word",
        ),
        pytest.param(
            "multilateral funding",
            None,
            {_IDS: ("multilateral",)},
            None,
            id="another-shape-codes-the-word-alone",
        ),
        pytest.param(
            "imputed multilateral ODA",
            _IMPUTED,
            {_IMPUTED: ("imputed multilateral oda",)},
            None,
            id="the-imputed-code-holds-the-word",
        ),
        pytest.param("ODA", "no-party-set", {}, [_CRS, "OECDDAC2A", _IDS], id="no-party-reading"),
    ],
)
def test_a_party_reading_stands_where_a_binding_shape_reads_all_any_other_reads(
    variable, creditor, readings, kept
) -> None:
    """Of the shapes recall offers "multilateral ... to Kenya", a DAC2a shape binds the
    multilaterals total as its provider and so reads the word. Where it reads every phrase
    the others read, it alone is the pool. The reading is withdrawn (`kept` is `None`) where
    a CRS shape reads a sector DAC2a does not, where the word stands beside one DAC2a does
    not read so no shape binds, and where another shape reads the word as a value of its
    own: by a graph label, as a World Bank IDS creditor type, or in a phrase of its codes,
    as the imputed multilateral aid type. A part with no party set keeps every shape."""
    state = _with_multilaterals(_state_naming(variable, "country"))
    if creditor == "no-party-set":
        state.party_sets = {}
    if creditor == _IMPUTED:
        rival = dataclasses.replace(_flow_shape(_IMPUTED, "attributedAmount"), shape_id=_IMPUTED)
    else:
        rival = _debt_shape(creditor if creditor and creditor.endswith("creditors") else None)
    shapes = [
        dataclasses.replace(_index_member_shape(_CRS_FLOW), shape_id=_CRS),
        dataclasses.replace(_index_member_shape(_DAC2A_ODA), shape_id="OECDDAC2A"),
        rival,
    ]

    def phrases_read(shape: ShapeDraft) -> frozenset[str]:
        return frozenset(readings.get(shape.shape_id, ()))

    if kept is None:
        with pytest.raises(PartyReadingWithdrawn):
            with_party_reading(state, shapes, phrases_read)
        return
    assert [s.shape_id for s in with_party_reading(state, shapes, phrases_read)] == kept


def test_a_pinned_recipient_keeps_its_kind() -> None:
    """A pinned place's set is named by its graph label and its resolution by the caller's
    text, so the set's kind is read by its node."""
    state = _state_naming("bilateral ODA", "country")
    (resolution,) = state.rcl.resolutions.values()
    state.rcl = Recall(candidates=[], resolutions={"KEN": resolution})

    assert _dataset_stated_words(_flow_shape("OECDDAC2A"), get_index(), state) == _STATED


def test_a_part_holding_no_declared_word_binds_no_place(monkeypatch) -> None:
    """DAC2a's stated word depends on the part's recipient, which is read from the places
    only when the part holds the word."""

    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("bound the part's places")

    monkeypatch.setattr(place_roles, "bind_place_roles", refuse)
    monkeypatch.setattr(place_roles, "assign_named_places", refuse)
    monkeypatch.setattr(phase_rank_bind, "parties_of_kind", refuse)
    shape = _flow_shape("OECDDAC2A")

    assert _dataset_stated_words(shape, get_index(), _state_naming("ODA", "country")) == _NOTHING
    with pytest.raises(AssertionError, match="bound the part's places"):
        _dataset_stated_words(shape, get_index(), _state_naming("bilateral ODA", "country"))


@pytest.mark.parametrize(
    ("roles", "names_and_roles", "places", "refusal"),
    [
        pytest.param(
            ("financingRecipient",),
            {"Kenya": ["origin"]},
            {"Kenya": "country/KEN"},
            "role_incompatible",
            id="origin-place-on-a-recipient-only-shape",
        ),
        pytest.param(
            ("creditor", "debtor"),
            {},
            {"Kenya": "country/KEN"},
            "role_not_resolved",
            id="one-undirected-place-on-two-required-roles",
        ),
        pytest.param(
            (),
            {"Kenya": ["destination"], "Norway": ["origin"]},
            {"Kenya": "country/KEN", "Norway": _NOR},
            "role_incompatible",
            id="both-flow-ends-on-a-subject-only-shape",
        ),
    ],
)
def test_places_a_shape_cannot_take_refuse(roles, names_and_roles, places, refusal) -> None:
    """Each refusal the elected shape answers as no-data drops an alternate instead."""
    shape = _shape_for(_economic_activity_arcs("Nominal", {}, roles))
    state = _state(
        names_and_roles={
            name: [_door_role(axis) for axis in axes] for name, axes in names_and_roles.items()
        },
        places=places,
    )

    assert bind_place_roles(state, shape) == refusal


def _index_member_shape(dcid: str) -> ShapeDraft:
    """The shape `derive_shapes` builds when recall confirms the fixture index row `dcid`
    alone, with the roles and constraints the row declares."""
    row = next(r for r in get_index().statvars if r.dcid == dcid)
    arcs = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "populationType": {"nodes": [{"dcid": row.population_type}]},
        "measuredProperty": {"nodes": [{"dcid": row.measured_property}]},
        "statType": {"nodes": [{"dcid": row.stat_type}]},
        "measurementQualifier": {"nodes": [{"dcid": row.measurement_qualifier}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in row.constraint_properties]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in row.constraint_properties.items()},
        "observationProperties": {"nodes": [{"dcid": r} for r in row.observation_properties]},
    }
    nodes = {dcid: {"label": dcid, "type": "StatisticalVariable", "arcs": arcs}}
    (shape,) = derive_shapes(
        confirmed_svs=[dcid], graph=FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
    )
    return shape


@pytest.mark.parametrize(
    ("variable", "member", "providers"),
    [
        pytest.param(
            "multilateral ODA",
            "Amount_FinancialFlow_Gross_Disbursement_DAC2aAidType206_Nominal",
            [_MULTILATERALS],
            id="dac2a-oda",
        ),
        pytest.param(
            "multilateral OOF",
            "Amount_FinancialFlow_Net_Disbursement_DAC2bAidType972_Nominal",
            [_MULTILATERALS],
            id="dac2b-oof",
        ),
        pytest.param(
            "multilateral ODA",
            "Amount_FinancialFlow_Net_Disbursement_DAC2bAidType972_Nominal",
            [],
            id="dac2b-reads-no-oda",
        ),
        pytest.param(
            "multilateral development bank ODA",
            "Amount_FinancialFlow_Gross_Disbursement_DAC2aAidType206_Nominal",
            [],
            id="beside-an-unread-word",
        ),
        pytest.param(
            "multilateral education aid",
            "Amount_FinancialFlow_Gross_Disbursement_DAC2aAidType206_Nominal",
            [],
            id="beside-a-sector",
        ),
        pytest.param(
            "multilateral aid for education",
            "Amount_FinancialFlow_Gross_Disbursement_DAC2aAidType206_Nominal",
            [_MULTILATERALS],
            id="beside-the-aid-noun",
        ),
        pytest.param(
            "total multilateral funding",
            "Amount_FinancialFlow_Gross_Disbursement_DAC2aAidType206_Nominal",
            [_MULTILATERALS],
            id="beside-amount-words",
        ),
        pytest.param(
            "multilateral ODA",
            "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
            "CRSPurposeCode11110_Nominal",
            [],
            id="crs-flow",
        ),
    ],
)
def test_an_index_shape_binds_the_party_its_family_datasets_declare(
    variable, member, providers
) -> None:
    """Before candidacy stamps its datasets, a shape recall confirmed from the fixture
    index answers from the datasets publishing its family: DAC2a's binds the multilaterals
    total as the provider of "multilateral ODA to Kenya", DAC2b's of "multilateral OOF to
    Kenya", and the CRS's none. The word names the party only beside words the shape reads
    (`place_roles._beside_words_read`): a DAC2b shape reads no "ODA", and no DAC2a shape
    reads "development" or "education"."""
    state = _with_multilaterals(_state_naming(variable, "country"))

    assignment = bind_place_roles(state, _index_member_shape(member))

    assert not isinstance(assignment, str)
    assert [s.node.dcid for s in assignment.sets.get("origin", ()) if s.node] == providers
    assert [s.name for s in assignment.sets["destination"]] == ["Kenya"]


@pytest.mark.parametrize(
    ("member", "lacks"),
    [
        pytest.param(
            "Amount_FinancialFlow_Gross_Disbursement_DAC1AidType1010_Nominal",
            True,
            id="dac1-provider-total",
        ),
        pytest.param(
            "Amount_FinancialFlow_Net_Disbursement_DAC1AidType8863_Nominal",
            True,
            id="dac1-flow-to-the-west-bank",
        ),
        pytest.param(
            "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
            "CRSPurposeCode11110_Nominal",
            False,
            id="crs-flow",
        ),
    ],
)
def test_an_index_shape_declares_the_axes_its_dataset_declares(member, lacks) -> None:
    """ "aid to Kenya" needs the recipient. DAC1 holds one on two flows that are twins of no
    provider total, so it declares none (`family_declares_roles`) and the DAC1 shape lacks
    the axis whichever member recall confirmed. The CRS shape, whose rows read with nothing
    bound hold a recipient, does not."""
    state = _state(
        names_and_roles={"Kenya": [_door_role("destination")]}, places={"Kenya": "country/KEN"}
    )

    assert lacks_query_axis(state, _index_member_shape(member)) is lacks


_CLIMATE_FINANCE = "undata/sdg/SERIES-DC_FIN_CLIMT"
_MOBILISED_RECEIPTS = "undata/sdg/SERIES-DC_OSSD_MPF"
_UNDECLARED_SERIES = "undata/sdg/SERIES-DC_FIN_UNDECLARED"
_JPN = "country/JPN"
_BGD = "country/BGD"
_PRIVATE_FINANCE_MOBILISED = "Test_AmountMobilised_Provider"


def _un_series_node(series: str, name: str) -> dict:
    """A UN series' one variable, whose population type is the series."""
    return {
        "label": name,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "populationType": {"nodes": [{"dcid": series, "name": name}]},
            "measuredProperty": {"nodes": [{"dcid": "value"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "constraintProperties": {"nodes": []},
        },
    }


def _un_series_shape(series: str) -> ShapeDraft:
    variable = series.replace("SERIES-", "")
    nodes = {variable: _un_series_node(series, series)}
    (shape,) = derive_shapes(
        confirmed_svs=[variable], graph=FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
    )
    return shape


@pytest.mark.parametrize(
    ("series", "axes", "lacks"),
    [
        pytest.param(_CLIMATE_FINANCE, ("destination",), True, id="provided-to-a-recipient"),
        pytest.param(_CLIMATE_FINANCE, ("origin",), False, id="provided-by-a-provider"),
        pytest.param(_MOBILISED_RECEIPTS, ("origin",), True, id="received-from-a-provider"),
        pytest.param(_MOBILISED_RECEIPTS, ("destination",), False, id="received-by-a-recipient"),
        pytest.param(_CLIMATE_FINANCE, ("origin", "destination"), False, id="both-ends"),
        pytest.param(_UNDECLARED_SERIES, ("destination",), False, id="undeclared"),
    ],
)
def test_a_one_way_series_lacks_the_flow_end_its_declared_side_opposes(
    series: str, axes: tuple[str, ...], lacks: bool
) -> None:
    """A UN series declares the end of a one-way flow its place sits on (the origin
    registry's `side`): climate finance provided sits on the provider, receipts of mobilised
    private finance on the recipient. A query that directs its places only to the other
    end asks for an axis the series cannot hold. One naming both ends keeps the binder's
    reading, and an undeclared series lacks no axis."""
    places = dict(list({"Kenya": "country/KEN", "Norway": _NOR}.items())[: len(axes)])
    state = _state(
        names_and_roles={
            name: [_door_role(axis, name)] for name, axis in zip(places, axes, strict=True)
        },
        places=places,
    )

    assert lacks_query_axis(state, _un_series_shape(series)) is lacks


def _resolve_directed(
    query: str, nodes: dict[str, dict], order: list[str], place: str, dcid: str, axis: str
):
    """Resolve `query` over `nodes`, recall returning `order` and `place` resolving to
    `dcid` on `axis`, every variable holding data at `dcid`."""
    graph = FakeGraph(
        nodes={**nodes, dcid: {"label": place, "type": "Country"}},
        obs={
            f"{sv}|{dcid}": [{"earliestDate": "2020", "latestDate": "2022", "obsCount": 3}]
            for sv in order
        },
        detect={query: {"svs": order, "cosine_scores": [0.8 - 0.1 * i for i in range(len(order))]}},
        resolve={place: dcid},
    )
    return asyncio.run(
        resolve_variable(
            query,
            entities=[place],
            date_request=None,
            detect_query=query,
            role_query=query,
            entity_roles={place: [DoorRole(axis=cast(Any, axis), span=place)]},
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_ElectFirstWithAlternate(),
            sdmx=FakeSdmxClient(observations={}, scope_availability=graph.scope_availability),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )


@pytest.mark.parametrize(
    ("series", "status", "reason"),
    [
        pytest.param(_CLIMATE_FINANCE, "no_data", "role_incompatible", id="declared-provider"),
        pytest.param(_UNDECLARED_SERIES, "definite", None, id="undeclared"),
    ],
)
def test_climate_finance_to_a_recipient_refuses_on_the_provider_side_series(
    series: str, status: str, reason: str | None
) -> None:
    """ "climate finance to Bangladesh": the only climate finance measure is the SDG series
    of finance provided, which has no recipients, so the part refuses `role_incompatible`
    in place of reading Bangladesh as a provider. A series with no declared side still
    binds Bangladesh as its subject and discloses the dropped direction."""
    variable = series.replace("SERIES-", "")
    nodes = {variable: _un_series_node(series, "Total climate-specific financial support provided")}

    result = _resolve_directed(
        "climate finance to Bangladesh", nodes, [variable], "Bangladesh", _BGD, "destination"
    )

    assert (result.status, result.no_data_reason) == (status, reason)
    assert ("direction_discarded" in [w.code for w in result.warnings]) is (status == "definite")


def test_private_finance_mobilised_by_a_provider_elects_the_provider_side_measure() -> None:
    """ "private sector mobilised by Japan's aid": the SDG series of mobilised private finance
    received by developing countries recalls first, and the binder elects the first shape on
    its menu. The series sits on the recipient, so it leaves the menu and the measure that
    declares a provider is elected."""
    variable = _MOBILISED_RECEIPTS.replace("SERIES-", "")
    nodes = {
        variable: _un_series_node(
            _MOBILISED_RECEIPTS,
            "Gross receipts by developing countries of mobilised private finance (MPF)",
        ),
        _PRIVATE_FINANCE_MOBILISED: {
            "label": "Private finance mobilised",
            "type": "StatisticalVariable",
            "arcs": {
                "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
                "populationType": {"nodes": [{"dcid": "PrivateFinanceMobilisation"}]},
                "measuredProperty": {"nodes": [{"dcid": "amountMobilised"}]},
                "statType": {"nodes": [{"dcid": "measuredValue"}]},
                "constraintProperties": {"nodes": []},
                "observationProperties": {"nodes": [{"dcid": "financingProvider"}]},
            },
        },
    }

    result = _resolve_directed(
        "private sector mobilised by Japan's aid",
        nodes,
        [variable, _PRIVATE_FINANCE_MOBILISED],
        "Japan",
        _JPN,
        "origin",
    )

    elected = [*result.specs, *(result.flagged_match or ())]
    assert [[m.ref.ref for m in spec.resolved.members] for spec in elected] == [
        [_PRIVATE_FINANCE_MOBILISED]
    ]


_GHA_KEN_GROUP = {"Africa": [_GHA, "country/KEN"]}


@pytest.mark.parametrize(
    ("roles", "over", "place", "elected_origin", "bound"),
    [
        pytest.param(
            ("creditor", "debtor"),
            "origin",
            "Kenya",
            False,
            {"destination": ["country/KEN"], "origin": []},
            id="rank-creditors-of-a-country-reads-it-as-debtor",
        ),
        pytest.param(
            ("financingProvider", "financingRecipient"),
            "destination",
            "Africa",
            False,
            "role_not_resolved",
            id="rank-recipients-in-a-group-is-ambiguous",
        ),
        pytest.param(
            ("creditor", "debtor"),
            "origin",
            "Africa",
            False,
            "role_not_resolved",
            id="rank-creditors-of-a-group-is-ambiguous",
        ),
        pytest.param(
            ("financingProvider", "financingRecipient"),
            "origin",
            "Kenya",
            True,
            {"destination": ["country/KEN"], "origin": []},
            id="operation-axis-outranks-the-elected-binding",
        ),
    ],
)
def test_an_operation_over_one_flow_axis_places_an_undirected_place(
    roles, over, place, elected_origin, bound
) -> None:
    """A rank over one flow axis of a two-role shape leaves that axis open, and a single
    place takes the other axis, also on an alternate whose elected shape bound the place on
    the ranked axis. A group can scope the ranked axis ("top recipients of aid in Africa")
    or name the other ("top creditors of external debt of Africa"), so the elected shape
    cannot tell its axis."""
    shape = _shape_for(_economic_activity_arcs("Nominal", {}, roles))
    groups = _GHA_KEN_GROUP if place in _GHA_KEN_GROUP else {}
    places = {place: "country/KEN"} if not groups else {}
    state = _state(names_and_roles={}, places=places)
    state.entities = [place]
    state.place_sets = place_sets(places, groups=groups)
    state.operations = (Operation(verb="rank", phrase="top", over=over),)
    if elected_origin:
        state.shape_draft = _shape_for(
            _economic_activity_arcs("Nominal", {}, ("financingProvider",))
        )
        elected = bind_place_roles(state, state.shape_draft)
        assert not isinstance(elected, str)
        assert settled_place_roles(state.shape_draft, elected).origin_dcids == ["country/KEN"]
        state.elected_place_roles = elected

    assigned = bind_place_roles(state, shape)

    if isinstance(bound, str):
        assert assigned == bound
    else:
        assert not isinstance(assigned, str), assigned
        roles_bound = settled_place_roles(shape, assigned)
        assert {
            "destination": roles_bound.destination_dcids,
            "origin": roles_bound.origin_dcids,
        } == bound


def test_a_region_beside_a_rank_over_one_flow_axis_is_read_as_a_group() -> None:
    """A continent named with no direction holds members, so a rank over the recipients
    does not send it to the origin axis as it does a single place: like a group it can
    scope the ranked axis or name the other."""
    shape = _shape_for(
        _economic_activity_arcs("Nominal", {}, ("financingProvider", "financingRecipient"))
    )
    state = _state(names_and_roles={}, places={})
    state.entities = ["Africa"]
    state.place_sets = place_sets(regions={"Africa": ("africa", [_GHA, "country/KEN"])})
    state.operations = (Operation(verb="rank", phrase="top", over="destination"),)

    assert bind_place_roles(state, shape) == "role_not_resolved"


@pytest.mark.parametrize(
    ("names_and_roles", "resolved", "role_bearing_outranks", "refused"),
    [
        pytest.param(
            {"Kenya": ["destination"], "Norway": ["origin"]},
            {"Kenya": "country/KEN", "Norway": _NOR},
            True,
            True,
            id="both-flow-ends",
        ),
        pytest.param(
            {"Norway": ["origin", "destination"]},
            {"Norway": _NOR},
            True,
            True,
            id="self-flow",
        ),
        pytest.param(
            {"Kenya": ["destination"]},
            {"Kenya": "country/KEN"},
            True,
            False,
            id="one-flow-end",
        ),
        pytest.param({}, {"Kenya": "country/KEN"}, True, False, id="no-flow-end"),
        pytest.param(
            {"Kenya": ["destination"], "Norway": ["origin"]},
            {"Kenya": "country/KEN"},
            True,
            False,
            id="origin-unresolved",
        ),
        pytest.param(
            {"Kenya": ["destination"], "Norway": ["origin"]},
            {"Kenya": "country/KEN", "Norway": _NOR},
            False,
            False,
            id="no-role-bearing-shape-outranks",
        ),
    ],
)
def test_subject_shapes_are_refused_only_when_both_resolved_flow_ends_are_named(
    names_and_roles, resolved, role_bearing_outranks, refused
) -> None:
    """The predicate that drops subject-only shapes from election matches the binder's
    refusal of every subject-only shape."""
    state = _state(
        names_and_roles={
            name: [_door_role(axis) for axis in axes] for name, axes in names_and_roles.items()
        },
        places=resolved,
    )
    state.entities = list(names_and_roles) or list(resolved)
    state.role_bearing_outranks = role_bearing_outranks
    subject_shape = _shape_for(_economic_activity_arcs("Nominal", {}))

    assert subject_shapes_refused(state) is refused
    assert (bind_place_roles(state, subject_shape) == "role_incompatible") is refused


_MODEL_WHERE = SlotBindingDraft(axis="where", property_dcid=None, kind="value", value_dcids=[_NOR])
_ACTIVITY_SOURCE = SlotBindingDraft(
    axis="how", property_dcid="activitySource", kind="value", value_dcids=["GrossNationalIncome"]
)


@pytest.mark.parametrize(
    ("places", "expected"),
    [
        pytest.param(
            {"Ghana": _GHA},
            [_ACTIVITY_SOURCE, where_binding([_GHA])],
            id="resolved-place-replaces-it",
        ),
        pytest.param({}, [_ACTIVITY_SOURCE], id="no-place-drops-it"),
    ],
)
def test_elected_shape_drops_a_model_emitted_entity_only_where_binding(places, expected) -> None:
    """The elected shape's where slot holds only the places it binds: a model-emitted where
    binding without a property gives way to the resolved place, or is dropped with none."""
    state = _state(names_and_roles={}, places=places)
    state.rcl = cast(Any, SimpleNamespace(resolutions={}))
    state.shape_draft = _shape_for(
        _economic_activity_arcs("PurchasingPowerParity", {"activitySource": "GrossNationalIncome"})
    )
    state.capped_shapes = [state.shape_draft]
    state.bind_result = elected_reply(0, bindings=[], also_plausible=[])
    state.elected_index = 0
    # With no place named, the shape's default scope reads Earth's countries.
    state.graph = FakeGraph(
        nodes={}, obs={}, detect={}, resolve={}, children={"Earth": {"Country": [_GHA]}}
    )
    state.sdmx = FakeSdmxClient(observations={}, scope_availability=state.graph.scope_availability)
    state.bindings = [_MODEL_WHERE, _ACTIVITY_SOURCE]

    assert asyncio.run(_phase_place_bindings(state)) is None
    assert state.bindings == expected


def test_placeless_election_reads_presence_for_its_flagged_alternates_too() -> None:
    """With no place named, one presence read covers elected GNI PPP and its flagged Atlas
    alternate, and the state keeps it for candidate admission. GNI PPP takes the default
    scope of its own StatVar, Ghana, and its scope warning is installed with it."""
    atlas = "Amount_EconomicActivity_GrossNationalIncome_AtlasMethod"
    nodes = {
        _GNI_PPP: _NODES[_GNI_PPP],
        atlas: {
            "label": "Gross national income, Atlas method",
            "type": "StatisticalVariable",
            "arcs": _economic_activity_arcs(
                "AtlasMethod", {"activitySource": "GrossNationalIncome"}
            ),
        },
    }
    graph = FakeGraph(
        nodes=nodes,
        obs={f"{_GNI_PPP}|{_GHA}": _GHA_OBS, f"{atlas}|{_NOR}": _GHA_OBS},
        detect={},
        resolve={},
        children={"Earth": {"Country": [_GHA, _NOR, _USA]}},
    )
    shapes = {
        next(iter(shape.sv_arc_facts or {})): shape
        for shape in derive_shapes(confirmed_svs=[_GNI_PPP, atlas], graph=graph)
    }
    state = _state(names_and_roles={}, places={})
    state.rcl = cast(Any, SimpleNamespace(resolutions={}))
    state.graph = graph
    state.sdmx = FakeSdmxClient(observations={}, scope_availability=graph.scope_availability)
    state.capped_shapes = [shapes[_GNI_PPP], shapes[atlas]]
    state.shape_draft = shapes[_GNI_PPP]
    state.bind_result = elected_reply(0, bindings=[], also_plausible=[1])
    state.elected_index = 0

    assert asyncio.run(_phase_place_bindings(state)) is None
    assert state.default_presence is not None
    assert {sv: frozenset(scope.places) for sv, scope in state.default_presence.scopes.items()} == {
        _GNI_PPP: frozenset({_GHA}),
        atlas: frozenset({_NOR}),
    }
    assert (state.place.mode, state.place.probe_dcids) == ("every_place", (_GHA,))
    assert [w.code for w in state.warnings if w.code != "membership_unverified"] == [
        "place_scope_defaulted"
    ]


@pytest.mark.parametrize(
    "order", [[_GNI_IDS, _GNI_PPP], [_GNI_PPP, _GNI_IDS]], ids=["ids_elected", "ppp_elected"]
)
def test_each_candidates_read_plan_reads_its_own_place_role(monkeypatch, order) -> None:
    """Whichever shape is elected, each candidate's read plan, the one its coverage and
    preview read, keys Ghana on that candidate's own role: the IDS series' recipient, GNI
    PPP's subject."""
    plans: dict[str, ReadPlan | None] = {}
    real = candidate_collapse.ground_candidates

    def _spy(state, candidate_triples):
        grounded = real(state, candidate_triples)
        for spec in grounded.specs:
            plans[spec.resolved.members[0].ref.ref] = grounded.cand_plans.get(spec.spec_id)
        return grounded

    monkeypatch.setattr(candidate_collapse, "ground_candidates", _spy)

    result = _resolve("gross national income of Ghana", order, [0.75, 0.74], ["Ghana"])

    assert result.status == "candidates", result
    role_tuples = {
        sv: [s.role_tuple for s in plan.series]
        for sv, plan in plans.items()
        if isinstance(plan, PairPlan)
    }
    assert role_tuples == {
        _GNI_IDS: [(("financingRecipient", _GHA),)],
        _GNI_PPP: [(("observationAbout", _GHA),)],
    }


def _scenario_candidates(monkeypatch, scenario_id: str, search_settings, **changes: Any):
    """Resolve a first-answer scenario, with `changes` to it, recording where each alternate
    is materialised and each grounded candidate's spec, probe entities, place context and
    bulk read."""
    scenario = dataclasses.replace(scenarios._SCENARIOS[scenario_id], **changes)
    scenarios._patch_settings(scenario, search_settings)
    materialised: dict[str, tuple[list[str] | None, dict[str, list[str]]]] = {}
    real_materialise = candidates.materialise

    def _materialise(shape, bindings, entities, **kwargs):
        materialised[next(iter(shape.sv_arc_facts))] = (kwargs["place_dcids"], dict(entities))
        return real_materialise(shape, bindings, entities, **kwargs)

    grounded_by_sv: dict[str, Any] = {}
    real_ground = candidate_collapse.ground_candidates

    def _ground(state, cands):
        grounded = real_ground(state, cands)
        for spec in grounded.specs:
            grounded_by_sv[spec.resolved.members[0].ref.ref] = SimpleNamespace(
                spec=spec,
                probe_entities=grounded.cand_probe_entities[spec.spec_id],
                place=grounded.cand_places[spec.spec_id],
                bulk_scope=grounded.cand_bulk_scope.get(spec.spec_id),
            )
        return grounded

    monkeypatch.setattr(candidates, "materialise", _materialise)
    monkeypatch.setattr(candidate_collapse, "ground_candidates", _ground)
    return scenarios._resolve(scenario), materialised, grounded_by_sv


def _place_warnings(result) -> list[tuple[str, str | None]]:
    return [
        (w.code, w.spec_id)
        for w in result.warnings
        if w.code in {"direction_discarded", "place_scope_defaulted"}
    ]


@pytest.mark.parametrize(
    ("scenario_id", "status"),
    [
        pytest.param("income-discovered-recipients-subject-alternate-at-origin", "candidates"),
        pytest.param("income-discovered-recipients-subject-alternate-at-recipient", "definite"),
    ],
    ids=["ppp-at-norway", "ppp-at-recipients"],
)
def test_subject_alternate_beside_discovered_recipients_is_probed_at_its_own_place(
    monkeypatch, search_settings, scenario_id, status
) -> None:
    """For "gross national income from Norway", the by-creditor IDS series discovers its
    recipients. GNI PPP reads Norway as its subject, so its availability is read at Norway,
    never at those recipients: it is offered where it has data at Norway, and where it has
    data only at a recipient it has none where the question asks and is never read
    (status rule 2). Offered, it is materialised and grounded at Norway and reports the
    direction it dropped, stamped with its spec id."""
    result, materialised, grounded = _scenario_candidates(monkeypatch, scenario_id, search_settings)

    assert result.status == status, result
    if status == "definite":
        assert scenarios._PPP not in materialised
        assert scenarios._PPP not in grounded
        assert _place_warnings(result) == []
        return
    assert materialised[scenarios._PPP] == (None, {"observationAbout": [scenarios._NOR]})
    ppp = grounded[scenarios._PPP]
    assert ppp.spec.definition.identity.entities == {"observationAbout": [scenarios._NOR]}
    assert ppp.probe_entities == {"observationAbout": [scenarios._NOR]}
    assert _place_warnings(result) == [("direction_discarded", ppp.spec.spec_id)]


def test_subject_alternate_beside_a_role_bearing_election_is_probed_over_every_place_it_binds(
    monkeypatch, search_settings
) -> None:
    """For "gross national income of Ghana from Norway", GNI PPP reads both places as its
    subject beside the IDS series that binds Ghana as recipient and Norway as provider. It is
    materialised and grounded over both, its coverage counts both countries, and it reports
    the direction it dropped, stamped with its spec id."""
    result, materialised, grounded = _scenario_candidates(
        monkeypatch, "income-role-bearing-elected-subject-alternate-at-both", search_settings
    )

    both = [scenarios._GHA, scenarios._NOR]
    assert result.status == "candidates", result
    assert materialised[scenarios._PPP] == (both, {"observationAbout": both})
    ppp = grounded[scenarios._PPP]
    assert ppp.probe_entities == {"observationAbout": both}
    assert ppp.place.probe_dcids == tuple(both)
    countries = next(
        d for d in ppp.spec.resolved.coverage.dimensions or () if d.label == "countries"
    )
    assert countries.count == 2
    assert _place_warnings(result) == [("direction_discarded", ppp.spec.spec_id)]


def test_placeless_subject_alternate_beside_a_role_bearing_election_takes_a_default_scope(
    monkeypatch, search_settings
) -> None:
    """With no place named, GNI PPP beside the elected IDS series is probed over the default
    scope a presence read gives it (Ghana), not over the IDS series' discovered recipient,
    and is offered with its own scope warning stamped with its spec id. The elected shape's
    role scope warning stays unstamped."""
    result, materialised, grounded = _scenario_candidates(
        monkeypatch, "placeless-role-bearing-elected-subject-alternate", search_settings
    )

    assert result.status == "candidates", result
    assert materialised[scenarios._PPP] == ([scenarios._GHA], {})
    ppp = grounded[scenarios._PPP]
    assert ppp.probe_entities == {"observationAbout": [scenarios._GHA]}
    assert (ppp.place.mode, ppp.place.probe_dcids) == ("every_place", (scenarios._GHA,))
    assert _place_warnings(result) == [("place_scope_defaulted", ppp.spec.spec_id)]
    assert [(w.code, w.spec_id) for w in result.warnings if w.code == "role_scope_defaulted"] == [
        ("role_scope_defaulted", None)
    ]


def test_placeless_subject_alternate_reads_its_default_scope_in_bulk(
    monkeypatch, search_settings
) -> None:
    """GNI PPP's default scope beside the elected IDS series holds two countries, so its
    preview reads the scope in bulk; the IDS series' own places read per pair."""
    result, _, grounded = _scenario_candidates(
        monkeypatch,
        "placeless-role-bearing-elected-subject-alternate",
        search_settings,
        obs=scenarios._at(scenarios._PPP, scenarios._GHA, scenarios._NOR),
    )

    assert result.status == "candidates", result
    assert grounded[scenarios._PPP].bulk_scope.places == (scenarios._GHA, scenarios._NOR)
    assert grounded[scenarios._IDS].bulk_scope is None


def test_each_candidates_slot_option_probe_binds_its_own_entities(
    monkeypatch, search_settings
) -> None:
    """Streamed slot-option probes bind each candidate's own entities: GNI PPP both places as
    its subject, and the IDS shape, narrowed to its by-creditor member since the query binds
    both roles, Ghana as recipient and Norway as provider."""
    probes: dict[str, dict[str, list[str]]] = {}

    def _gather(batch, **_: Any):
        for _spec_id, shape, _bindings, _options, entities, *_ in batch:
            probes[next(iter(shape.sv_arc_facts))] = entities
        return asyncio.sleep(0, result=[])

    monkeypatch.setattr(candidate_options, "_gather_slot_options_events", _gather)
    monkeypatch.setattr(candidate_options, "create_owned_task", asyncio.ensure_future)
    result, _, _ = _scenario_candidates(
        monkeypatch,
        "income-role-bearing-elected-subject-alternate-at-both",
        search_settings,
        stream_slot_options=True,
    )

    assert result.status == "candidates", result
    assert probes == {
        scenarios._IDS_BC: {
            "financingProvider": [scenarios._NOR],
            "financingRecipient": [scenarios._GHA],
        },
        scenarios._PPP: {"observationAbout": [scenarios._GHA, scenarios._NOR]},
    }


def test_a_candidate_whose_place_warnings_match_the_shape_in_play_adds_none(
    monkeypatch, search_settings
) -> None:
    """With no place named, the flagged count's default scope holds other countries than the
    elected count's, in the same mode, so its scope warning equals the one already shown and
    nothing is stamped."""
    result, materialised, _ = _scenario_candidates(
        monkeypatch, "placeless-flagged-alternate-elsewhere", search_settings
    )

    assert result.status == "candidates", result
    assert materialised[scenarios._FAR][0] == [scenarios._TZA]
    assert _place_warnings(result) == [("place_scope_defaulted", None)]


def _request_spans(monkeypatch, scenario_id: str, search_settings) -> VirtualRequests:
    """The requests of one scenario's resolve, each with its span on a virtual clock."""
    scenario = scenarios._SCENARIOS[scenario_id]
    scenarios._patch_settings(scenario, search_settings)
    requests = VirtualRequests()
    requests.install(
        monkeypatch,
        doubles=(FakeGraph, FakeSdmxClient),
        # The stamp, the backends and the UN reader are read without a request.
        free=("generation_stamp", "source_key", "un_availability"),
        pools=scenarios._POOL_MODULES,
        executors=scenarios._EXECUTORS,
    )
    scenarios._resolve(scenario, requests.run)
    return requests


def test_placeless_alternate_presence_read_runs_beside_the_elected_materialisation(
    monkeypatch, search_settings
) -> None:
    """GNI PPP's default scope needs Earth's countries and an availability read beside the
    elected, role-bearing IDS series. The countries are listed before the elected shape is
    materialised, beside its walk, and the availability read starts with that
    materialisation, so neither adds a step in series."""
    spans = _request_spans(
        monkeypatch,
        "placeless-role-bearing-elected-subject-alternate-at-recipient",
        search_settings,
    )

    _, listing_end = spans.first("child_places_batch")
    read_start, _ = spans.first("availability_values")
    elected_start, _ = spans.first("default_facet_policy")
    assert listing_end <= elected_start
    assert read_start == elected_start


def test_a_placeless_election_reads_its_siblings_presence_in_its_own_read(
    monkeypatch, search_settings
) -> None:
    """The estimated count's StatVar rides the elected count's scope read, whose
    availability reads end before the elected count is materialised, so a zero-observation
    re-election reads no scope of its own and adds no step in series."""
    reads = _availability_reads(monkeypatch)
    spans = _request_spans(monkeypatch, "placeless-reelect-adopt", search_settings)

    assert sorted(reads) == sorted([scenarios._MORT, scenarios._MORT_EST])
    read_end = max(end for name, _, _, end in spans.spans if name == "availability_values")
    elected_start, _ = spans.first("default_facet_policy")
    assert read_end <= elected_start


def test_placeless_siblings_with_separate_default_scopes_are_read_side_by_side(
    monkeypatch, search_settings
) -> None:
    """The estimated and the projected count hold data in different countries, so each is
    probed over its own default scope. Their facet reads start together, so a second scope
    adds no step in series."""
    spans = _request_spans(monkeypatch, "placeless-reelect-two-siblings-apart", search_settings)

    estimated_start, _ = spans.first("observation_facets_batch", [scenarios._MORT_EST])
    projected_start, _ = spans.first("observation_facets_batch", [scenarios._MORT_PROJ])
    assert estimated_start == projected_start


def _availability_reads(monkeypatch) -> list[str]:
    """Record the StatVar of each default-scope availability read, in call order."""
    real = ScopeAvailability.availability_values
    reads: list[str] = []

    def _read(self, *, variables, **kwargs):
        reads.extend(variables)
        return real(self, variables=variables, **kwargs)

    monkeypatch.setattr(ScopeAvailability, "availability_values", _read)
    return reads


def _place_checks(
    monkeypatch, stat_vars: Collection[str], on_check: Callable[[], None]
) -> tuple[list[list[str]], list[str]]:
    """Run `on_check` in each default-scope check of listed places (`node_arcs_many` over
    `typeOf`) that covers a place one of `stat_vars` holds data at. Returns the StatVars whose
    places each such check covered and the thread each ran on: the thread that reads the
    scope, once the availability reads it waits for have answered."""
    real = FakeGraph.node_arcs_many
    checks: list[list[str]] = []
    threads: list[str] = []

    def _check(self, dcids, *, properties, deadline=None):
        if tuple(properties) == ("typeOf",):
            places = set(dcids)
            covered = sorted({sv for sv, place in self.scope_availability.pairs if place in places})
            if set(covered) & set(stat_vars):
                checks.append(covered)
                threads.append(threading.current_thread().name)
                on_check()
        return real(self, dcids, properties=properties, deadline=deadline)

    monkeypatch.setattr(FakeGraph, "node_arcs_many", _check)
    return checks, threads


def _presence_reads_failing_for(
    monkeypatch,
    stat_var: str,
    message: str,
    *,
    error: type[Exception] = GraphInfraError,
    beside: Collection[str] = (),
) -> list[list[str]]:
    """Fail each default-scope check of listed places covering `stat_var`, or a StatVar its
    scope is read `beside`, with `error`, and return the StatVars each such check covered."""

    def _fail() -> None:
        raise error(message)

    checks, _threads = _place_checks(monkeypatch, {stat_var, *beside}, _fail)
    return checks


def test_no_presence_read_is_made_for_an_alternate_the_admission_guards_reject(
    monkeypatch,
) -> None:
    """With no place named, the IDS series is elected beside a flagged mortality count that
    reads another population as its subject, which the admission guards reject. No
    presence read names it, so a read that would fail for it cannot fail the request, and
    the IDS series answers as it does without the alternate."""
    reads = _presence_reads_failing_for(
        monkeypatch, scenarios._MORT, "the mortality count's presence read failed"
    )
    scenario = scenarios._Scenario(
        scenarios._GNI,
        (),
        (scenarios._IDS, scenarios._MORT),
        (0.85, 0.84),
        scenarios._at(scenarios._MORT, scenarios._KEN),
        also_plausible=(1,),
        coverage_rows=((scenarios._IDS, "f_ids", "", scenarios._USA, scenarios._YEARS),),
    )

    result = scenarios._resolve(scenario)

    assert result.status == "definite", result
    (spec,) = result.specs
    assert [sv.ref.ref for sv in spec.resolved.members] == [scenarios._IDS]
    assert reads == []


@pytest.mark.parametrize(
    ("scenario_id", "read", "status"),
    [
        # The elected count holds data, so re-election does not run.
        ("placeless-definite-beside-sibling", [scenarios._MORT, scenarios._MORT_EST], "definite"),
    ],
)
def test_no_presence_read_is_made_for_a_shape_the_answer_does_not_reach(
    monkeypatch, search_settings, scenario_id, read, status
) -> None:
    """The siblings ride the elected shape's scope read. When the answer does not reach
    them, no other availability read is sent."""
    with monkeypatch.context() as patch:
        reads = _availability_reads(patch)
        _request_spans(patch, scenario_id, search_settings)

    assert sorted(reads) == sorted(read)
    result = scenarios._resolve(scenarios._SCENARIOS[scenario_id])
    assert result.status == status, result


def test_more_requests_than_walk_workers_keep_every_alternate_evaluated(monkeypatch) -> None:
    """Four requests run at once over one walk worker, each walk taking a while. Every walk
    waits its turn in the walk pool's queue: each request offers GNI PPP beside the IDS
    series, as one request alone does, none states `alternates_unevaluated`, every walk runs
    on the walk pool and every presence read on the presence reads' pool, and none on the
    default executor."""
    _await_a_free_alternates_presence_pool()
    scenario = scenarios._SCENARIOS["placeless-role-bearing-elected-subject-alternate"]
    alone = scenarios._resolve(scenario)
    threads: dict[str, list[str]] = {"walk": [], "presence": []}
    walk = phase_place_materialise.walk_alternate
    presence = phase_place_materialise._presence_after_listing

    def _slow_walk(**kwargs):
        threads["walk"].append(threading.current_thread().name)
        time.sleep(0.05)
        return walk(**kwargs)

    def _presence(**kwargs):
        threads["presence"].append(threading.current_thread().name)
        return presence(**kwargs)

    monkeypatch.setattr(
        phase_place_materialise,
        "_WALK_POOL",
        concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="alternate-walks"),
    )
    monkeypatch.setattr(phase_place_materialise, "walk_alternate", _slow_walk)
    monkeypatch.setattr(phase_place_materialise, "_presence_after_listing", _presence)

    async def _four_requests() -> list[Any]:
        def _resolve(resolve: Coroutine[Any, Any, Any]) -> Any:
            return resolve

        return await asyncio.wait_for(
            asyncio.gather(*[scenarios._resolve(scenario, _resolve) for _ in range(4)]),
            timeout=10,
        )

    answers = asyncio.run(_four_requests())

    assert [scenarios._answer_record(answer) for answer in answers] == [
        scenarios._answer_record(alone)
    ] * 4
    assert all(
        "alternates_unevaluated" not in [w.code for w in answer.warnings] for answer in answers
    )
    assert len(threads["walk"]) >= 4
    assert all(name.startswith("alternate-walks") for name in threads["walk"]), threads
    assert threads["presence"]
    assert all(name.startswith("alternates-reads") for name in threads["presence"]), threads


def _stalling_presence_reads(monkeypatch, stat_var: str) -> tuple[threading.Event, list[str]]:
    """Make every default-scope check of listed places covering `stat_var` wait until the
    returned event is set, recording the thread each one runs on. The check runs on the
    thread that reads the scope, so a stalled check holds that thread."""
    release = threading.Event()
    _checks, threads = _place_checks(monkeypatch, {stat_var}, release.wait)
    return release, threads


def _await_a_free_alternates_presence_pool() -> None:
    """Wait until every worker of the alternates' presence pool is free, as a read an
    earlier test released finishes."""
    slots = phase_place_materialise._ALTERNATES_SLOTS
    workers = phase_place_materialise._ALTERNATES_WORKERS
    for _ in range(workers):
        assert slots.acquire(timeout=5), "an alternates presence read is still running"
    for _ in range(workers):
        slots.release()


def test_a_stalled_alternates_presence_read_holds_no_worker_required_work_runs_on(
    monkeypatch, caplog
) -> None:
    """On an event loop whose default executor has two workers, requests electing the IDS
    series with no place named start the presence read GNI PPP needs beside its
    materialisation until every worker of the alternates' pool holds one, and the presence
    source stalls. Two more such requests then find no worker free, at the start or at
    candidate admission: they make no read and answer with the IDS series alone, stating
    that another reading was not evaluated. A request naming Kenya answers too. Every alternate's read runs on the alternates' pool, none on
    the default executor. Once the source answers, the first requests answer as well and
    the pool is free again."""
    _await_a_free_alternates_presence_pool()
    release, threads = _stalling_presence_reads(monkeypatch, scenarios._PPP)
    workers = phase_place_materialise._ALTERNATES_WORKERS
    role_elected = scenarios._SCENARIOS["placeless-role-bearing-elected-subject-alternate"]
    named = scenarios._SCENARIOS["named-definite"]

    def _resolve(resolve: Coroutine[Any, Any, Any]) -> Any:
        return resolve

    async def _requests() -> tuple[list[Any], list[Any]]:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=2))
        waiting = [
            asyncio.ensure_future(scenarios._resolve(role_elected, _resolve))
            for _ in range(workers)
        ]
        try:
            async with asyncio.timeout(5):
                while len(threads) < workers:
                    await asyncio.sleep(0.01)
                answered = await asyncio.gather(
                    scenarios._resolve(role_elected, _resolve),
                    scenarios._resolve(role_elected, _resolve),
                    scenarios._resolve(named, _resolve),
                )
            assert not any(task.done() for task in waiting)
        finally:
            release.set()
        return answered, await asyncio.wait_for(asyncio.gather(*waiting), timeout=5)

    with caplog.at_level(logging.WARNING, logger=phase_place_materialise.__name__):
        answered, role_answers = asyncio.run(_requests())

    without_alternate = scenarios._answer_record(
        scenarios._resolve(scenarios._SCENARIOS["placeless-role-bearing-definite"])
    )
    assert [scenarios._answer_record(result) for result in answered[:2]] == [
        without_alternate,
        without_alternate,
    ]
    assert answered[2].status == "definite", answered[2]
    assert [result.status for result in role_answers] == ["candidates"] * workers
    assert len(threads) == workers
    assert all(name.startswith("alternates-reads") for name in threads), threads
    assert all(_states_unevaluated(result, "no_capacity") for result in answered[:2])
    left_out = [r for r in caplog.records if "left unevaluated" in r.getMessage()]
    assert len(left_out) == 2
    assert all("(no_capacity)" in r.getMessage() for r in left_out)
    _await_a_free_alternates_presence_pool()


def test_an_alternates_presence_read_the_answer_does_not_reach_is_not_waited_for(
    monkeypatch,
) -> None:
    """The IDS series' materialisation fails while the presence read GNI PPP would need is
    stalled. The failure reaches the caller at once: nothing waits for the read."""
    _await_a_free_alternates_presence_pool()
    release, threads = _stalling_presence_reads(monkeypatch, scenarios._PPP)

    def _fails(*args: Any, **kwargs: Any) -> Any:
        # Fail only once the presence read holds its thread, so the read is in flight.
        deadline = time.monotonic() + 5
        while not threads and time.monotonic() < deadline:
            time.sleep(0.01)
        raise GraphInfraError("the elected shape's materialisation failed")

    monkeypatch.setattr(phase_place_materialise, "run_materialise", _fails)

    async def _answer() -> Any:
        resolve = scenarios._resolve(
            scenarios._SCENARIOS["placeless-role-bearing-elected-subject-alternate"],
            lambda resolve: resolve,
        )
        try:
            return await asyncio.wait_for(resolve, timeout=5)
        finally:
            release.set()

    with pytest.raises(GraphInfraError, match="materialisation failed"):
        asyncio.run(_answer())
    assert len(threads) == 1


def _alternates_presence_workers_busy(monkeypatch, *, refusals: int | None) -> None:
    """Find no worker of the alternates' pool free for the first `refusals` presence reads,
    or for every presence read when `refusals` is `None`."""
    real = phase_place_materialise._submit_alternates_read
    left = [refusals]

    def submit(read, /, **kwargs):
        if read is phase_place_materialise._presence_after_listing:
            if left[0] is None:
                return None
            if left[0]:
                left[0] -= 1
                return None
        return real(read, **kwargs)

    monkeypatch.setattr(phase_place_materialise, "_submit_alternates_read", submit)


_REASONS = ("no_capacity", "upstream_error", "timed_out")


def _alternates_presence_read_incomplete(
    monkeypatch, reason: str, stat_var: str, *, beside: Collection[str] = ()
) -> list[list[str]]:
    """Keep the presence read naming `stat_var` from completing, for `reason`: no worker of
    the alternates' pool is free, the presence source fails, or it times out. Returns the
    presence reads sent."""
    if reason == "no_capacity":
        _alternates_presence_workers_busy(monkeypatch, refusals=None)
        return _presence_reads_failing_for(monkeypatch, stat_var, "no read is made for it")
    # A check the request's deadline cuts ends the scope read in a timeout; a check past its
    # own window would fall back to countries instead.
    error = TimeoutError if reason == "timed_out" else GraphInfraError
    return _presence_reads_failing_for(
        monkeypatch, stat_var, f"{reason}: the read", error=error, beside=beside
    )


_UNEVALUATED_CAUSES = {
    "no_capacity": "because the service was at capacity",
    "upstream_error": "because a data source failed",
    "timed_out": "because a data source timed out",
}


def _states_unevaluated(result: Any, reason: str) -> bool:
    """Whether the answer states, once and in plain words, that another reading of the
    question was not evaluated (`alternates_unevaluated`), giving the cause `reason` stands
    for."""
    stated = [
        (w.severity, w.message) for w in result.warnings if w.code == "alternates_unevaluated"
    ]
    if not stated:
        return False
    assert stated == [
        (
            "info",
            "Another reading of this question could not be checked "
            f"{_UNEVALUATED_CAUSES[reason]}, so this answer may not be the only one.",
        )
    ], stated
    return True


def _admission_reasons(monkeypatch) -> list[dict[int, str]]:
    """Record, for each candidate admission, the reason each unevaluated alternate carries
    on the admission result (`_AlternatesAdmission.unevaluated`)."""
    real = candidate_admission.admit_alternates
    reasons: list[dict[int, str]] = []

    async def _admit(state: Any) -> Any:
        admission = await real(state)
        reasons.append(dict(admission.unevaluated))
        return admission

    monkeypatch.setattr(candidate_admission, "admit_alternates", _admit)
    return reasons


def _without_unevaluated(projection: dict[str, Any]) -> dict[str, Any]:
    """`projection` without its `alternates_unevaluated` warning."""
    warnings = [w for w in projection["warnings"] if w[0] != "alternates_unevaluated"]
    return {**projection, "warnings": warnings}


def test_with_no_alternates_presence_worker_free_at_the_start_admission_reads_on_the_pool(
    monkeypatch, search_settings
) -> None:
    """With no worker of the alternates' presence pool free beside the IDS series'
    materialisation, the read GNI PPP needs is not started there. A worker is free by
    candidate admission, which makes the read once on that pool, never on the default
    executor: Earth's countries were listed beside the elected shape's walk, so its presence
    read adds one step in series, and the answer is the one given with a worker free."""
    scenario_id = "placeless-role-bearing-elected-subject-alternate"
    with monkeypatch.context() as patch:
        free = _request_spans(patch, scenario_id, search_settings)
    free_answer = scenarios._resolve(scenarios._SCENARIOS[scenario_id])
    with monkeypatch.context() as patch:
        _alternates_presence_workers_busy(patch, refusals=1)
        reads = _availability_reads(patch)
        spans = _request_spans(patch, scenario_id, search_settings)
    _alternates_presence_workers_busy(monkeypatch, refusals=1)
    release, threads = _stalling_presence_reads(monkeypatch, scenarios._PPP)
    release.set()
    answer = scenarios._resolve(scenarios._SCENARIOS[scenario_id])

    assert reads == [scenarios._PPP]
    assert spans.serial_depth() == free.serial_depth() + 1
    assert scenarios._answer_record(answer) == scenarios._answer_record(free_answer)
    assert len(threads) == 1
    assert threads[0].startswith("alternates-reads"), threads


def _untraced(value: Any) -> Any:
    """`value` without the resolve traces it holds, which count every shape recalled."""
    if isinstance(value, dict):
        return {key: _untraced(item) for key, item in value.items() if key != "trace"}
    if isinstance(value, list):
        return [_untraced(item) for item in value]
    return value


@pytest.mark.parametrize("reason", _REASONS)
@pytest.mark.parametrize(
    ("scenario", "without_alternate"),
    [
        (
            scenarios._SCENARIOS["placeless-role-bearing-elected-subject-alternate"],
            scenarios._SCENARIOS["placeless-role-bearing-definite"],
        ),
        (
            scenarios._SCENARIOS["placeless-role-bearing-elected-subject-alternate-at-recipient"],
            scenarios._SCENARIOS["placeless-role-bearing-definite"],
        ),
        # The IDS series holds no data.
        (
            scenarios._Scenario(
                scenarios._GNI,
                (),
                (scenarios._IDS, scenarios._PPP),
                (0.85, 0.84),
                scenarios._at(scenarios._PPP, scenarios._GHA),
                also_plausible=(1,),
            ),
            scenarios._Scenario(scenarios._GNI, (), (scenarios._IDS,), (0.85,)),
        ),
    ],
    ids=["ppp-elsewhere", "ppp-at-recipient", "ids-without-data"],
)
def test_an_unevaluated_alternate_leaves_the_elected_answer_standing_and_stated(
    monkeypatch, scenario, without_alternate, reason
) -> None:
    """The presence read GNI PPP needs does not complete: no worker of the alternates' pool
    is free, the presence source fails, or it times out. The answer is the one
    given when only the IDS series is found, with `alternates_unevaluated` added. The
    admission result carries the reason; PPP is not offered, the answer is marked as leaving
    an alternate out, and a read that failed is not sent again."""
    reads = _alternates_presence_read_incomplete(monkeypatch, reason, scenarios._PPP)
    reasons = _admission_reasons(monkeypatch)

    answer = scenarios._resolve(scenario)

    assert reads == ([] if reason == "no_capacity" else [[scenarios._PPP]])
    assert _states_unevaluated(answer, reason)
    assert reasons[0] == {1: reason}
    assert answer.alternates_left_out
    assert _untraced(_without_unevaluated(scenarios._projection(answer))) == _untraced(
        scenarios._projection(scenarios._resolve(without_alternate))
    )


@pytest.mark.parametrize("reason", ["timed_out", "upstream_error"])
def test_an_alternate_whose_presence_read_after_unbinding_fails_is_unevaluated(
    monkeypatch, reason
) -> None:
    """GNI PPP's presence is known, and materialising it unbinds values that match no member
    together; the presence read its relaxed bindings need does not complete
    (`candidates._presence_after_unbinding`). The reading is unevaluated, not found empty:
    the admission result carries the reason, and the answer states `alternates_unevaluated`
    for it and stays out of the cache."""
    reasons = _admission_reasons(monkeypatch)
    monkeypatch.setattr(
        candidate_admission,
        "_materialise_alt_shapes",
        lambda alternates, *args, presence, **kwargs: _AlternatesRead(
            [], None, presence, tuple((alternate, reason) for alternate in alternates)
        ),
    )

    answer = scenarios._resolve(
        scenarios._SCENARIOS["placeless-role-bearing-elected-subject-alternate"]
    )

    assert reasons[0] == {1: reason}
    assert _states_unevaluated(answer, reason)
    assert answer.alternates_left_out


class _GniRequestLLM(scenarios._Bind):
    """Extracts "gross national income" naming no place, then elects the IDS series and
    flags GNI PPP."""

    def __init__(self) -> None:
        super().__init__(0, (1,))

    async def generate_structured(self, *, prompt, system, schema):
        if schema is Extraction:
            return Extraction(
                kind="resolution",
                variables=[scenarios._GNI],
                entities=[],
                dates=[],
                dropped_qualifiers=[],
            ), None
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


def _resolve_gni_request(
    monkeypatch, cache: ResponseCache | None, *, trace_id: str = "0" * 32
) -> ResolveResponse:
    """Resolve the raw query "gross national income" through `resolve_async` over the graph
    of `placeless-role-bearing-elected-subject-alternate`, with `trace_id` bound as the app
    binds a request's."""
    scenario = scenarios._SCENARIOS["placeless-role-bearing-elected-subject-alternate"]
    patch_coverage(monkeypatch, scenario.coverage_rows, role_names=scenarios._ROLE_NAMES)
    graph = FakeGraph(
        nodes=scenarios._NODES,
        obs=dict(scenario.obs),
        detect={
            scenario.query: {"svs": list(scenario.svs), "cosine_scores": list(scenario.scores)}
        },
        resolve=scenarios._RESOLVE,
        places=scenarios._PLACES,
        children=scenarios._CHILDREN,
    )

    async def _resolve() -> ResolveResponse:
        token = set_trace_id(trace_id)
        try:
            return await resolve_async(
                ResolveRequest(input=RawTextInput(query=scenario.query)),
                graph=graph,
                llm=_GniRequestLLM(),
                sdmx=FakeSdmxClient(
                    observations={}, availability={}, scope_availability=graph.scope_availability
                ),
                # The mint's walk settles the members, so the answer may be cached.
                materialise_graph=scenarios._materialise_graph(graph, scenario),
                response_cache=cache,
                generation=UNKNOWN_GENERATION,
            )
        finally:
            reset_trace_id(token)

    with mock.patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
        return asyncio.run(_resolve())


def _untimed_parts(response: ResolveResponse) -> list[dict[str, Any]]:
    """The response's parts without the step timings they and their traces hold, which vary
    between runs."""
    return scenarios._without_timings(
        [part.model_dump(mode="json", exclude={"timing_ms"}) for part in response.parts]
    )


@pytest.mark.parametrize("reason", _REASONS)
def test_an_answer_that_left_an_alternate_unevaluated_is_not_cached(
    monkeypatch, caplog, reason
) -> None:
    """The presence read GNI PPP needs does not complete, so the raw query "gross national
    income" answers with the IDS series alone and states `alternates_unevaluated`. The log
    line naming the unevaluated shape carries the reason and the request's trace id. That answer is not stored: the next identical request, with the read
    completing, misses the cache and answers with both GNI series, as a request made
    without the cache does."""
    _await_a_free_alternates_presence_pool()
    cache = ResponseCache()
    trace_id = "1" * 32
    with monkeypatch.context() as incomplete:
        _alternates_presence_read_incomplete(incomplete, reason, scenarios._PPP)
        with caplog.at_level(logging.WARNING, logger=phase_place_materialise.__name__):
            left_out = _resolve_gni_request(monkeypatch, cache, trace_id=trace_id)

    answered = _resolve_gni_request(monkeypatch, cache)

    assert left_out.status == "definite"
    (part,) = left_out.parts
    assert _states_unevaluated(part, reason)
    assert answered.status == "candidates"
    assert _untimed_parts(answered) == _untimed_parts(_resolve_gni_request(monkeypatch, None))
    assert cache.stats().hits == 0
    logged = [r.getMessage() for r in caplog.records if "left unevaluated" in r.getMessage()]
    assert len(logged) == 1
    assert f"({reason})" in logged[0]
    assert logged[0].endswith(f"trace_id={trace_id}"), logged


def test_an_answer_that_left_no_alternate_out_is_cached(monkeypatch) -> None:
    """With a worker of the alternates' presence pool free, the raw query "gross national
    income" answers with both GNI series and the answer is stored: the next identical
    request is served from the cache, even with no worker free."""
    _await_a_free_alternates_presence_pool()
    cache = ResponseCache()
    answered = _resolve_gni_request(monkeypatch, cache)
    with monkeypatch.context() as busy:
        _alternates_presence_workers_busy(busy, refusals=None)
        replayed = _resolve_gni_request(monkeypatch, cache)

    assert answered.status == "candidates"
    assert replayed.parts == answered.parts
    assert cache.stats().hits == 1


@pytest.mark.parametrize("left_unevaluated", [True, False])
def test_every_pass_result_states_an_unevaluated_alternate_or_nothing(left_unevaluated) -> None:
    """A pass's result, a capped alternate's refusal included, states
    `alternates_unevaluated` exactly when the pass left an alternate unevaluated, and is
    left out of the cache exactly then."""
    refusal = pipeline.RegionResult(
        variable_text="gross national income",
        status="unsupported",
        specs=(),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
    )
    state = SimpleNamespace(
        unevaluated_alternates={1: "timed_out"} if left_unevaluated else {},
        alternates_presence_failure="timed_out",
    )

    stated = pipeline._stating_unevaluated(refusal, cast(Any, state))

    assert _states_unevaluated(stated, "timed_out") == left_unevaluated
    assert stated.alternates_left_out == left_unevaluated


def test_a_failed_presence_read_the_elected_shape_needs_fails_the_request(monkeypatch) -> None:
    """With no place named, the elected mortality count takes its default scope from its own
    scope read. That read is required: a graph failure of its check of listed places
    reaches the caller as a graph error and no answer is served."""
    reads = _presence_reads_failing_for(
        monkeypatch, scenarios._MORT, "the elected shape's presence read failed"
    )

    with pytest.raises(GraphInfraError, match="the elected shape's presence read failed"):
        scenarios._resolve(scenarios._SCENARIOS["placeless-definite-beside-sibling"])
    assert len(reads) == 1


def test_a_re_election_menu_is_materialised_once(monkeypatch, search_settings) -> None:
    """Re-election offers the estimated and the projected count in place of the empty
    elected one; the flagged alternate rides on no refusal, since the siblings answer
    (status rule 1). Candidate admission takes both siblings as re-election materialised
    them and materialises nothing more, so each offered shape is materialised once and each
    sibling's facets are read twice: once by re-election and once for the facet options its
    candidate offers."""
    scenario_id = "placeless-reelect-menu-beside-flagged-alternate-together"
    materialised: list[str] = []
    real_materialise = candidates.materialise

    def _materialise(shape, bindings, entities, **kwargs):
        materialised.append(next(iter(shape.sv_arc_facts)))
        return real_materialise(shape, bindings, entities, **kwargs)

    monkeypatch.setattr(candidates, "materialise", _materialise)

    spans = _request_spans(monkeypatch, scenario_id, search_settings)

    assert materialised == [scenarios._MORT_EST, scenarios._MORT_PROJ]
    facet_reads = [
        stat_vars for name, stat_vars, _, _ in spans.spans if name == "observation_facets_batch"
    ]
    for stat_var in (scenarios._MORT_EST, scenarios._MORT_PROJ):
        assert facet_reads.count([stat_var]) == 2, facet_reads


def test_a_role_bearing_elected_shape_reads_no_presence_for_its_siblings(
    monkeypatch, search_settings
) -> None:
    """A role-bearing elected shape on a query naming no place discovers its scope. At the
    declared default price basis, its subject-only siblings differ from it only on that
    closed axis and on role class, so they are neither rivals nor stand-ins (status rules 1
    and 2): no presence read and no listing of Earth's countries is sent for them, with or
    without data on the elected shape."""
    nominal_ids = scenarios._sv(
        scenarios._NODES[scenarios._IDS]["label"],
        scenarios._income_arcs(
            "Nominal", {"economicActivityType": "GrossNationalIncome"}, ("financingRecipient",)
        ),
    )
    for scenario_id in (
        "placeless-role-bearing-elected-subject-siblings",
        "placeless-role-bearing-elected-with-data-subject-siblings",
    ):
        with monkeypatch.context() as patch:
            patch.setitem(scenarios._NODES, scenarios._IDS, nominal_ids)
            spans = _request_spans(patch, scenario_id, search_settings)
        names = {name for name, _, _, _ in spans.spans}
        assert not names & {
            "availability_values",
            "entities_with_data_by_variable",
            "child_places_batch",
        }, names


class _Captured(Exception):
    """Ends an option probe once the entities it binds are captured."""


def test_each_option_probe_of_a_lone_candidate_binds_its_own_entities(monkeypatch) -> None:
    """Slot-option enrichment, entity options and the slot-option relaunch of a candidate
    bind the entities of its own place context, not those of the shape in play."""
    own = {"observationAbout": [_GHA, _NOR]}
    spec = SimpleNamespace(
        spec_id="spec_ppp",
        definition=SimpleNamespace(identity=SimpleNamespace(facets=None)),
        resolved=SimpleNamespace(members=()),
    )
    grounded = SimpleNamespace(
        cand_places={
            "spec_ppp": SimpleNamespace(
                roles=SimpleNamespace(resolved_entities=own, open_axes=frozenset())
            )
        },
        cand_probe_inputs={"spec_ppp": (object(), [], [], [_GHA, _NOR])},
        cand_role_axis_facts={"spec_ppp": (object(), [object()], [_GHA, _NOR])},
        cand_obs_entities={"spec_ppp": ([_GNI_PPP], own)},
        cand_entity_options_raw={},
    )
    state = SimpleNamespace(
        place=SimpleNamespace(
            roles=SimpleNamespace(resolved_entities={"financingRecipient": [_GHA]})
        ),
        include_option_coverage=True,
        stream_slot_options=True,
        date_request=None,
        materialise_graph=None,
        entity_objects=[],
        graph=None,
        sdmx=None,
    )
    seen: list[Any] = []

    def _capture_entities(*args: Any, **kwargs: Any):
        seen.append(kwargs.get("entities", args[1] if len(args) > 1 else None))
        raise _Captured

    monkeypatch.setattr(candidate_options, "_maybe_enrich_slot_options", _capture_entities)
    monkeypatch.setattr(candidate_options, "_build_and_enrich_entity_options", _capture_entities)
    monkeypatch.setattr(candidate_options, "_candidate_facet_probe", lambda *_: None)

    with pytest.raises(_Captured):
        asyncio.run(
            candidate_options._attach_candidate_slot_options(
                cast(Any, state), cast(Any, grounded), cast(Any, spec)
            )
        )
    with pytest.raises(_Captured):
        asyncio.run(
            candidate_options._attach_candidate_entity_options(
                cast(Any, state), cast(Any, grounded), cast(Any, spec)
            )
        )
    relaunch = candidate_options._make_slot_options_relaunch(
        cast(Any, state), cast(Any, grounded), cast(Any, spec)
    )

    assert relaunch is not None
    assert [*seen, relaunch.keywords["entities"]] == [own, own, own]
