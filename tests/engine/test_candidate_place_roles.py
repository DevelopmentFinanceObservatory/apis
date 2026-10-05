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
from collections.abc import Callable, Collection, Coroutine
from datetime import date
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from dfo_platform.trace import reset_trace_id, set_trace_id
from graph_generation import UNKNOWN_GENERATION
from materialise.graph import Facet as MaterialiseFacet
from materialise.graph import SubjectPage
from materialise.place_scope import NO_DEFAULT_SCOPE, DefaultPresence
from search.engine import assemble
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.core import resolve_async
from search.engine.errors import GraphInfraError
from search.engine.extract import Extraction
from search.engine.region import (
    candidate_admission,
    candidate_collapse,
    candidate_options,
    candidates,
    phase_answer,
    phase_place_materialise,
    pipeline,
)
from search.engine.region.phase_place_materialise import _phase_place_bindings
from search.engine.region.pipeline import resolve_variable
from search.engine.region.place_roles import (
    bind_place_roles,
    subject_shapes_refused,
    where_binding,
)
from search.engine.region.state import ResolveState
from search.engine.response_cache import ResponseCache
from search.engine.retrieval.derive import derive_shapes
from search.engine.retrieval.read_plan import PairPlan, ReadPlan
from search.engine.retrieval.retrieve import UnsupportedDraft
from search.engine.role_text import DoorRole
from search.engine.shape import ShapeDraft
from search.models import (
    DatasetResubmitInput,
    OptionsFor,
    OptionsRequest,
    RawTextInput,
    ResolveRequest,
    ResolveResponse,
    Warning,
)
from series_coverage.schema import RoleNames
from tests.engine import test_first_answer_place_scenarios as scenarios
from tests.engine._fake_coverage import (
    FakeCoverageService,
    build_coverage_index,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query, offline_resolve
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


_NODES = {
    _GNI_IDS: {
        "label": "Gross national income (current US$)",
        "type": "StatisticalVariable",
        "arcs": _economic_activity_arcs(
            "Nominal",
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
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[], also_plausible=[1]), None


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
            base_steps=[],
            base_timing={},
            coverage=_coverage().snapshot(),
        )
    )


def _spec_serving(result, stat_var: str):
    specs = [
        spec
        for spec in result.specs
        if [item.ref for item in spec.resolved.stat_vars.items] == [stat_var]
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
        qualifier="Nominal",
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
    assert [item["value"]["ref"]["dcid"] for item in response.json()["options"]["items"]] == [_GHA]


class _ElectFirstOnly:
    """Elects the first shape on the menu and names no alternate."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[], also_plausible=[]), None


def test_reelected_subject_only_sibling_answers_under_its_own_place_roles() -> None:
    """The IDS series is elected but has data only at Norway, so its subject-only sibling
    GNI PPP is re-elected for Ghana. The definite spec binds Ghana as GNI PPP's subject, and
    its identity resubmits to the same spec."""
    query = "gross national income of Ghana"
    order, scores = [_GNI_IDS, _GNI_PPP], [0.85, 0.84]
    graph = _graph(query, order, scores)
    coverage = FakeCoverageService(
        build_coverage_index([(_GNI_IDS, "f_ids", "", _NOR, _YEARS)], role_names=_ROLE_NAMES)
    )
    result = asyncio.run(
        resolve_variable(
            query,
            entities=["Ghana"],
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_ElectFirstOnly(),
            sdmx=FakeSdmxClient(observations={}, scope_availability=graph.scope_availability),
            base_steps=[],
            base_timing={},
            coverage=coverage.snapshot(),
        )
    )

    assert result.status == "definite", result
    (spec,) = result.specs
    assert [item.ref for item in spec.resolved.stat_vars.items] == [_GNI_PPP]
    assert "elected_shape_no_data" in [w.code for w in result.warnings]
    assert spec.definition.identity.entities == {"observationAbout": [_GHA]}
    assert [entity.role for entity in spec.resolved.entities] == ["observationAbout"]

    replay = offline_resolve(
        ResolveRequest(
            input=DatasetResubmitInput(entries=[_identity_entry_from_spec(spec, query)])
        ),
        graph=_graph(query, order, scores),
        sdmx=FakeSdmxClient(observations={}, scope_availability=graph.scope_availability),
    )
    assert replay.parts[0].status == "definite", replay.parts[0]
    assert replay.parts[0].spec.spec_id == spec.spec_id


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
            "Nominal",
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
            base_steps=[],
            base_timing={},
            coverage=coverage.snapshot(),
        )
    )


def test_reelected_sibling_probes_reads_and_dates_every_subject_place(monkeypatch) -> None:
    """The IDS shape binds Ghana as its recipient and Norway as its provider but has no data
    there. GNI PPP is re-elected with both countries as its subjects, and its coverage, date
    range and observation read cover both, as its resubmit does. Norway's dropped direction
    warns once."""
    read_entities: list[dict[str, list[str]]] = []
    observation_entities = phase_answer._observation_entities

    def recording_observation_entities(shape, **kwargs):
        entities_read, warning = observation_entities(shape, **kwargs)
        read_entities.append(entities_read)
        return entities_read, warning

    monkeypatch.setattr(phase_answer, "_observation_entities", recording_observation_entities)
    result = _resolve_mixed()

    assert result.status == "definite", result
    (spec,) = result.specs
    assert [item.ref for item in spec.resolved.stat_vars.items] == [_GNI_PPP]
    codes = [w.code for w in result.warnings]
    assert codes.count("direction_discarded") == 1, codes
    assert "elected_shape_no_data" in codes
    assert spec.definition.identity.entities == {"observationAbout": [_GHA, _NOR]}
    assert read_entities == [{"observationAbout": [_GHA, _NOR]}]
    coverage = spec.resolved.coverage
    assert coverage.kind == "exact"
    assert coverage.observation_count == 56
    assert {"label": "countries", "count": 2} in [
        dim.model_dump() for dim in coverage.dimensions or []
    ]
    (item,) = spec.resolved.stat_vars.items
    assert (item.data_date_range.start, item.data_date_range.end) == ("1990", "2024")

    replay = offline_resolve(
        ResolveRequest(
            input=DatasetResubmitInput(entries=[_identity_entry_from_spec(spec, _MIXED_QUERY)])
        ),
        graph=_mixed_graph(),
        sdmx=FakeSdmxClient(observations={}),
    )
    assert replay.parts[0].status == "definite", replay.parts[0]
    assert replay.parts[0].spec.spec_id == spec.spec_id
    assert replay.parts[0].spec.resolved.coverage == coverage


def test_reelected_siblings_read_plan_reads_both_subject_places(monkeypatch) -> None:
    """The re-elected GNI PPP answer reads the plan its own scope built: one series at each
    subject place, Ghana and Norway, the plan its coverage, headline and preview read."""
    plans: list[ReadPlan | None] = []
    real = phase_answer.filter_answer

    def recording_filter_answer(mat, *args, **kwargs):
        plans.append(mat.read_plan)
        return real(mat, *args, **kwargs)

    monkeypatch.setattr(phase_answer, "filter_answer", recording_filter_answer)
    result = _resolve_mixed()

    assert result.status == "definite", result
    (plan,) = plans
    assert isinstance(plan, PairPlan)
    assert plan.members == (_GNI_PPP,)
    assert sorted(s.role_tuple for s in plan.served) == [
        (("observationAbout", _GHA),),
        (("observationAbout", _NOR),),
    ]


def test_reelected_sibling_reports_its_own_carrier_warnings(monkeypatch) -> None:
    """The elected IDS carrier stands on unverified role declarations, but GNI PPP answers,
    so the answer carries no role-declaration warning."""
    run_materialise = phase_place_materialise.run_materialise

    def unverified_run_materialise(*args, **kwargs):
        return dataclasses.replace(
            run_materialise(*args, **kwargs), role_declaration_unverified=True
        )

    monkeypatch.setattr(phase_place_materialise, "run_materialise", unverified_run_materialise)
    result = _resolve_mixed()

    assert [item.ref for item in result.specs[0].resolved.stat_vars.items] == [_GNI_PPP]
    assert "role_declaration_unverified" not in [w.code for w in result.warnings]


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
            base_steps=[],
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


def test_siblings_probed_in_separate_scopes_join_the_menu_in_rank_order() -> None:
    """Three siblings of the empty IDS shape hold data: GNI PPP and the Atlas series probe
    Ghana and Norway as their subjects, and the constant-price IDS series, ranked between
    them, probes Ghana as its recipient. The candidates keep rank order across the two
    probe scopes."""
    order = [
        _GNI_IDS,
        _GNI_IDS_BY_CREDITOR,
        _GNI_PPP,
        _GNI_REAL_BY_CREDITOR,
        _GNI_REAL,
        _GNI_ATLAS,
    ]
    graph = FakeGraph(
        nodes={
            **_MIXED_NODES,
            _GNI_REAL_BY_CREDITOR: {
                "label": "Gross national income (constant US$), by creditor",
                "type": "StatisticalVariable",
                "arcs": _economic_activity_arcs(
                    "RealValue",
                    {
                        "economicActivityType": "GrossNationalIncome",
                    },
                    ("financingProvider", "financingRecipient"),
                ),
            },
            _GNI_REAL: {
                "label": "Gross national income (constant US$)",
                "type": "StatisticalVariable",
                "arcs": _economic_activity_arcs(
                    "RealValue",
                    {
                        "economicActivityType": "GrossNationalIncome",
                    },
                    ("financingRecipient",),
                ),
            },
            _GNI_ATLAS: {
                "label": "Gross national income, Atlas method",
                "type": "StatisticalVariable",
                "arcs": _economic_activity_arcs(
                    "AtlasMethod", {"activitySource": "GrossNationalIncome"}
                ),
            },
        },
        obs={f"{_GNI_PPP}|{_GHA}": _GHA_OBS, f"{_GNI_ATLAS}|{_GHA}": _GHA_OBS},
        detect={
            _MIXED_QUERY: {"svs": order, "cosine_scores": [0.85, 0.85, 0.84, 0.83, 0.83, 0.82]}
        },
        resolve={"Ghana": _GHA, "Norway": _NOR},
    )
    result = _resolve_electing_first(
        _MIXED_QUERY,
        _MIXED_ENTITIES,
        graph=graph,
        coverage_rows=[
            (_GNI_IDS, "f_ids", "", _USA, _YEARS),
            (_GNI_IDS_BY_CREDITOR, "f_ids", _USA, _USA, _YEARS),
            (_GNI_REAL_BY_CREDITOR, "f_real", _NOR, _GHA, _YEARS),
            (_GNI_REAL, "f_real", "", _GHA, _YEARS),
        ],
        role_names=_MIXED_ROLE_NAMES,
    )

    assert result.status == "candidates", result
    assert [[item.ref for item in spec.resolved.stat_vars.items] for spec in result.specs] == [
        [_GNI_PPP],
        [_GNI_REAL_BY_CREDITOR],
        [_GNI_ATLAS],
    ]


_TWO_RECIPIENTS_QUERY = "gross national income to Ghana and Kenya"


@pytest.mark.parametrize(
    ("query", "entities", "graph", "coverage_rows"),
    [
        pytest.param(
            _MIXED_QUERY,
            _MIXED_ENTITIES,
            _mixed_graph,
            [
                (_GNI_IDS, "f_ids", "", _USA, _YEARS),
                (_GNI_IDS_BY_CREDITOR, "f_ids", _USA, _USA, _YEARS),
            ],
            id="elected-scope-unclamped",
        ),
        pytest.param(
            _TWO_RECIPIENTS_QUERY,
            ["Ghana", "Kenya"],
            lambda: FakeGraph(
                nodes={**_NODES, _KEN: {"label": "Kenya", "type": "Country"}},
                obs={f"{_GNI_PPP}|{_GHA}": _GHA_OBS},
                detect={
                    _TWO_RECIPIENTS_QUERY: {
                        "svs": [_GNI_IDS, _GNI_PPP],
                        "cosine_scores": [0.85, 0.84],
                    }
                },
                resolve={"Ghana": _GHA, "Kenya": _KEN},
            ),
            [(_GNI_IDS, "f_ids", "", _NOR, _YEARS)],
            id="elected-scope-clamped",
        ),
    ],
)
def test_reelected_sibling_reports_its_own_probe_clamp(
    search_settings, query, entities, graph, coverage_rows
) -> None:
    """With one probe place allowed, re-elected GNI PPP unions two subject places and
    reports that clamp once: beside an IDS shape that probes Ghana alone and never clamps,
    and beside one whose own two-recipient clamp it replaces."""
    search_settings(max_standard_probe_places=1)
    result = _resolve_electing_first(
        query,
        entities,
        graph=graph(),
        coverage_rows=coverage_rows,
        role_names=_MIXED_ROLE_NAMES,
    )

    assert result.status == "definite", result
    assert [item.ref for item in result.specs[0].resolved.stat_vars.items] == [_GNI_PPP]
    assert [w.message for w in result.warnings if w.code == "standard_probe_places_clamped"] == [
        "2 probe places detected across 2 named place(s); resolved the first 1."
    ]


def test_reelected_sibling_place_warnings_follow_its_installation(search_settings) -> None:
    """The IDS series binds Ghana and Kenya as its recipients and clamps their union, then
    finds no data. Re-elected GNI PPP drops both directions and clamps the same union: its
    place warnings follow in its own order, directions then the clamp, and the re-election
    warning follows them."""
    search_settings(max_standard_probe_places=1)
    result = _resolve_electing_first(
        _TWO_RECIPIENTS_QUERY,
        ["Ghana", "Kenya"],
        graph=FakeGraph(
            nodes={**_NODES, _KEN: {"label": "Kenya", "type": "Country"}},
            obs={f"{_GNI_PPP}|{_GHA}": _GHA_OBS},
            detect={
                _TWO_RECIPIENTS_QUERY: {"svs": [_GNI_IDS, _GNI_PPP], "cosine_scores": [0.85, 0.84]}
            },
            resolve={"Ghana": _GHA, "Kenya": _KEN},
        ),
        coverage_rows=[(_GNI_IDS, "f_ids", "", _NOR, _YEARS)],
        role_names=_MIXED_ROLE_NAMES,
    )

    assert result.status == "definite", result
    # No member walk runs here, so the members answer unsettled (`membership_unverified`).
    assert [w.code for w in result.warnings] == [
        "membership_unverified",
        "direction_discarded",
        "direction_discarded",
        "standard_probe_places_clamped",
        "elected_shape_no_data",
    ]


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
    assert [item.ref for item in result.specs[0].resolved.stat_vars.items] == [_GNI_IDS]
    assert "place_scope_defaulted" not in [w.code for w in result.warnings]
    assert (["Earth"], "Country") not in enumerated


def test_alternate_whose_role_cannot_take_the_place_is_dropped(caplog) -> None:
    """Norway named as the origin binds as GNI PPP's subject, but the IDS series declares
    only a recipient, so it is dropped from the candidates and its binding failure is logged."""
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

    with caplog.at_level(logging.INFO, logger="search.engine.region.place_roles"):
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
                sdmx=FakeSdmxClient(observations={}, scope_availability=graph.scope_availability),
                base_steps=[],
                base_timing={},
                coverage=_coverage().snapshot(),
            )
        )

    assert result.status == "definite", result
    assert [item.ref for item in result.specs[0].resolved.stat_vars.items] == [_GNI_PPP]
    assert result.specs[0].definition.identity.entities == {"observationAbout": [_NOR]}
    unbound = [r.getMessage() for r in caplog.records if "cannot bind" in r.getMessage()]
    assert len(unbound) == 1
    assert "'economicactivity_amount_measuredvalue_nominal_by_role'" in unbound[0]
    assert "role_incompatible" in unbound[0]


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
        rcl=cast(Any, SimpleNamespace(resolved_entity_names=places)),
        standard_place_dcids=list(places.values()),
        role_bearing_recalled=True,
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
        source_text=None,
        materialise_graph=None,
        materialise_sdmx=None,
        pipeline_steps=[],
        timing={},
        warnings=[],
        coverage=empty_coverage_snapshot(),
    )


def _door_role(axis: str) -> DoorRole:
    return door_roles_for_test_query(
        f"aid {'from' if axis == 'origin' else 'to'} Kenya", ["Kenya"]
    )["Kenya"][0]


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


@pytest.mark.parametrize(
    ("names_and_roles", "resolved", "role_bearing_recalled", "refused"),
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
            id="no-role-bearing-shape-recalled",
        ),
    ],
)
def test_subject_shapes_are_refused_only_when_both_resolved_flow_ends_are_named(
    names_and_roles, resolved, role_bearing_recalled, refused
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
    state.role_bearing_recalled = role_bearing_recalled
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
    state.rcl = cast(Any, SimpleNamespace(resolved_entity_names=places, resolutions={}))
    state.shape_draft = _shape_for(
        _economic_activity_arcs("PurchasingPowerParity", {"activitySource": "GrossNationalIncome"})
    )
    state.capped_shapes = [state.shape_draft]
    state.bind_result = _UniversalBindOutput(chosen_shape_index=0, bindings=[], also_plausible=[])
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
    state.rcl = cast(Any, SimpleNamespace(resolved_entity_names={}, resolutions={}))
    state.graph = graph
    state.sdmx = FakeSdmxClient(observations={}, scope_availability=graph.scope_availability)
    state.capped_shapes = [shapes[_GNI_PPP], shapes[atlas]]
    state.shape_draft = shapes[_GNI_PPP]
    state.bind_result = _UniversalBindOutput(chosen_shape_index=0, bindings=[], also_plausible=[1])

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
            plans[spec.resolved.stat_vars.items[0].ref] = grounded.cand_plans.get(spec.spec_id)
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
            grounded_by_sv[spec.resolved.stat_vars.items[0].ref] = SimpleNamespace(
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
        if w.code
        in {"direction_discarded", "place_scope_defaulted", "standard_probe_places_clamped"}
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
    recipients. GNI PPP reads Norway as its subject, so it is materialised and grounded at
    Norway, never at those recipients: it is offered where it has data at Norway and dropped
    where it has data only at a recipient. Offered, it reports the direction it dropped,
    stamped with its spec id."""
    result, materialised, grounded = _scenario_candidates(monkeypatch, scenario_id, search_settings)

    assert result.status == status, result
    assert materialised[scenarios._PPP] == (None, {"observationAbout": [scenarios._NOR]})
    if status == "definite":
        assert scenarios._PPP not in grounded
        assert _place_warnings(result) == []
        return
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
    assert [sv.ref for sv in spec.resolved.stat_vars.items] == [scenarios._IDS]
    assert reads == []


@pytest.mark.parametrize(
    ("scenario_id", "read", "status"),
    [
        # The elected count holds data, so re-election does not run.
        ("placeless-definite-beside-sibling", [scenarios._MORT, scenarios._MORT_EST], "definite"),
        # The tied reading holds data, so the widened search does not run.
        (
            "placeless-tie-neighbour-has-data",
            [scenarios._MORT, scenarios._LIVESTOCK, scenarios._FAR],
            "candidates",
        ),
    ],
)
def test_no_presence_read_is_made_for_a_shape_the_answer_does_not_reach(
    monkeypatch, search_settings, scenario_id, read, status
) -> None:
    """The siblings and the widened search's shapes ride the elected shape's scope read.
    When the answer does not reach them, no other availability read is sent."""
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
    is free, the presence source fails, or it times out. No tie fired. The answer is the one
    given when only the IDS series is found, with `alternates_unevaluated` added. The
    admission result carries the reason; PPP is not offered, the answer is marked as leaving
    an alternate out, and a read that failed is not sent again."""
    reads = _alternates_presence_read_incomplete(monkeypatch, reason, scenarios._PPP)
    reasons = _admission_reasons(monkeypatch)

    answer = scenarios._resolve(scenario)

    assert reads == ([] if reason == "no_capacity" else [[scenarios._PPP]])
    assert _states_unevaluated(answer, reason)
    assert reasons[0] == {1: reason}
    assert "near_neighbor_ambiguous" not in [w.code for w in answer.warnings]
    assert answer.alternates_left_out
    assert _untraced(_without_unevaluated(scenarios._projection(answer))) == _untraced(
        scenarios._projection(scenarios._resolve(without_alternate))
    )


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


_TRADE = "Amount_GrossNationalTradeScenario"
_REMIT = "Amount_GrossNationalRemittanceScenario"


def _amount_reading(label: str, population: str) -> dict[str, Any]:
    return scenarios._sv(
        label,
        {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": label}]},
            "populationType": {"nodes": [{"dcid": population}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    )


def _tied_trade_scenario(monkeypatch, trade_places: tuple[str, ...]) -> scenarios._Scenario:
    """With no place named, the IDS series is elected in a near-neighbour tie with a
    subject-only trade reading holding data at `trade_places`; a remittance reading lies
    beyond the tie's margin."""
    monkeypatch.setitem(
        scenarios._NODES,
        _TRADE,
        _amount_reading("Gross national income of trade", "TradeActivityScenario"),
    )
    monkeypatch.setitem(
        scenarios._NODES,
        _REMIT,
        _amount_reading("Gross national remittance amount", "RemittanceActivityScenario"),
    )
    return scenarios._Scenario(
        scenarios._GNI,
        (),
        (scenarios._IDS, _TRADE, _REMIT, scenarios._MORT),
        (0.85, 0.849, 0.84, 0.83),
        {
            **scenarios._at(_TRADE, *trade_places),
            **scenarios._at(_REMIT, scenarios._GHA),
            **scenarios._at(scenarios._MORT, scenarios._KEN),
        },
        coverage_rows=((scenarios._IDS, "f_ids", "", scenarios._USA, scenarios._YEARS),),
    )


_TIE_UNRESOLVED = (
    "the chosen reading and a semantically distinct alternative scored closely enough to be "
    "ambiguous; the ambiguity stands because another reading could not be checked."
)


@pytest.mark.parametrize("reason", _REASONS)
@pytest.mark.parametrize(
    ("trade_places", "free_answer"),
    [
        # The tied trade reading holds data, so it is offered beside the IDS series.
        ((scenarios._KEN,), [scenarios._IDS, _TRADE]),
        # It holds none, so the search widens to the remittance reading.
        ((), [scenarios._IDS, _REMIT]),
    ],
)
def test_a_tied_reading_left_unevaluated_keeps_its_tie_on_the_elected_answer(
    monkeypatch, trade_places, free_answer, reason
) -> None:
    """With the tied reading's presence read completing, the query offers the tied or the
    widened reading as candidates. When that read does not complete, the tied reading's
    data is never read, which is no evidence against it. The IDS series is served alone
    with the Spec it has with no alternate, at the confidence a tied candidate has: medium
    in place of high. `near_neighbor_ambiguous` stays, saying the ambiguity stands, and
    `alternates_unevaluated` is added, with the reason on the admission result. The tied
    reading is not offered, no search
    widens past it, the answer is marked as leaving an alternate out, and a read that
    failed is not sent again."""
    scenario = _tied_trade_scenario(monkeypatch, trade_places)

    def _stat_vars(result: Any) -> list[str]:
        return [spec.resolved.stat_vars.items[0].ref for spec in result.specs]

    free = scenarios._resolve(scenario)
    assert free.status == "candidates", free
    assert _stat_vars(free) == free_answer
    assert "near_neighbor_ambiguous" in [w.code for w in free.warnings]
    assert not _states_unevaluated(free, reason)
    assert not free.alternates_left_out
    assert free.specs[0].resolved.confidence == "medium"

    # The tied reading's scope is read beside the remittance reading's, so a check of the
    # places either lists is that read's.
    reads = _alternates_presence_read_incomplete(monkeypatch, reason, _TRADE, beside=(_REMIT,))
    reasons = _admission_reasons(monkeypatch)
    answer = scenarios._resolve(scenario)
    without_alternate = scenarios._resolve(scenarios._SCENARIOS["placeless-role-bearing-definite"])

    assert len(reads) == (0 if reason == "no_capacity" else 1)
    assert answer.status == "definite", answer
    assert _stat_vars(answer) == [scenarios._IDS]
    assert not answer.flagged_match
    assert answer.alternate_match is None
    assert _states_unevaluated(answer, reason)
    assert reasons[0] == {1: reason}
    tie = [w for w in answer.warnings if w.code == "near_neighbor_ambiguous"]
    assert [(w.severity, w.message) for w in tie] == [("info", _TIE_UNRESOLVED)]
    assert "near_neighbor_pool_widened" not in [w.code for w in answer.warnings]
    assert answer.alternates_left_out
    (spec,) = answer.specs
    (elected,) = without_alternate.specs
    assert elected.resolved.confidence == "high"
    assert spec.resolved.confidence == "medium"
    assert spec.spec_id == elected.spec_id
    assert spec.definition == elected.definition
    assert _untraced(spec.resolved.model_dump(mode="json", exclude={"confidence"})) == _untraced(
        elected.resolved.model_dump(mode="json", exclude={"confidence"})
    )
    extra = [
        w
        for w in answer.warnings
        if w.code not in ("near_neighbor_ambiguous", "alternates_unevaluated")
    ]
    assert extra == list(without_alternate.warnings)


def test_a_candidates_part_keeps_a_tie_whose_other_reading_is_unevaluated() -> None:
    """A candidates part that serves fewer than two tied readings drops the tie's warning.
    When the part states an unevaluated alternate, the warning stays and says the ambiguity
    stands."""
    tie = Warning(code="near_neighbor_ambiguous", severity="info", message="provisional")
    unevaluated = Warning(code="alternates_unevaluated", severity="info", message="unread")

    assert assemble._with_final_near_neighbor_message([tie], [], frozenset()) == []
    assert assemble._with_final_near_neighbor_message([tie, unevaluated], [], frozenset()) == [
        tie.model_copy(update={"message": _TIE_UNRESOLVED}),
        unevaluated,
    ]


def _seeding_known_trade_presence(monkeypatch) -> None:
    """Give the request the trade reading's presence, without data anywhere, before the
    first candidate admission reads, as a read made earlier in the request would."""
    real = candidate_admission._read_alternates_presence
    seeded: list[bool] = []

    async def _seeded(state: Any, indices: Any, *, group: Any) -> Any:
        if not seeded:
            seeded.append(True)
            state.default_presence = DefaultPresence(
                countries=(), scopes={_TRADE: NO_DEFAULT_SCOPE}
            )
        return await real(state, indices, group=group)

    monkeypatch.setattr(candidate_admission, "_read_alternates_presence", _seeded)


@pytest.mark.parametrize("reason", ["upstream_error", "timed_out"])
@pytest.mark.parametrize("seeded", [False, True], ids=["control", "trade-presence-known"])
def test_a_failed_presence_read_is_followed_by_no_other(monkeypatch, reason, seeded) -> None:
    """The presence read over the tied trade reading and the remittance reading a widened
    search adds fails. With the trade reading's presence already known, that failure leaves
    no alternate unevaluated at first: the trade reading is read, holds no data and the
    search widens. The widened search sends no read of its own after the failure, as the
    control sends none: the remittance reading is unevaluated, and the answer says so."""
    scenario = _tied_trade_scenario(monkeypatch, ())
    reads = _alternates_presence_read_incomplete(monkeypatch, reason, _REMIT)
    if seeded:
        _seeding_known_trade_presence(monkeypatch)
    reasons = _admission_reasons(monkeypatch)

    answer = scenarios._resolve(scenario)

    assert len(reads) == 1
    assert answer.status == "definite", answer
    assert _states_unevaluated(answer, reason)
    assert answer.alternates_left_out
    assert reasons[0] == ({2: reason} if seeded else {1: reason})


def _gap_filled(monkeypatch, first: Any, second: Any) -> Any:
    """`first` after the second pass over the UN System's candidates answers `second`."""

    async def _settled() -> None:
        return None

    async def _second_pass(state: Any) -> Any:
        return second

    monkeypatch.setattr(pipeline, "_run_phases", _second_pass)
    monkeypatch.setattr(pipeline, "_log_gap_fill", lambda *args, **kwargs: None)
    rcl = SimpleNamespace(
        settle_un=_settled, un_unavailable=False, un_candidates=[object()], un_leg=None
    )
    state = SimpleNamespace(
        rcl=None, shape_draft=None, n_below_score_floor=0, n_measure_class_refused=0
    )
    return asyncio.run(
        pipeline._fill_gap(
            first,
            rcl=cast(Any, rcl),
            deadline=None,
            variable_index=0,
            emit=None,
            first_sentences=[],
            first_measure_class_refused=0,
            second_state=lambda emit, bound: cast(Any, state),
        )
    )


_IDS_WITHOUT_DATA = scenarios._Scenario(
    scenarios._GNI,
    (),
    (scenarios._IDS, scenarios._PPP),
    (0.85, 0.84),
    scenarios._at(scenarios._PPP, scenarios._GHA),
    also_plausible=(1,),
)


@pytest.mark.parametrize("unevaluated_pass", ["first", "second"])
def test_a_gap_filled_answer_states_an_alternate_either_pass_left_unevaluated(
    monkeypatch, unevaluated_pass
) -> None:
    """The IDS series holds no data and GNI PPP's presence read fails, so the pass ends
    `no_data` with PPP unevaluated. When that is the first pass, the second pass's answer
    is served; when it is the second pass, the first pass's `no_data` stands. Either way
    the served answer states `alternates_unevaluated` once and is left out of the cache,
    since a completed read could have changed it."""
    with monkeypatch.context() as failing:
        _alternates_presence_read_incomplete(failing, "upstream_error", scenarios._PPP)
        unevaluated = scenarios._resolve(_IDS_WITHOUT_DATA)
    assert unevaluated.status == "no_data", unevaluated
    assert _states_unevaluated(unevaluated, "upstream_error")
    if unevaluated_pass == "first":
        first = unevaluated
        second = scenarios._resolve(scenarios._SCENARIOS["placeless-role-bearing-definite"])
    else:
        first = scenarios._resolve(_IDS_WITHOUT_DATA)
        second = unevaluated
        assert first.status == "no_data", first
    assert _states_unevaluated(first, "upstream_error") == (unevaluated_pass == "first")

    served = _gap_filled(monkeypatch, first, second)

    assert served.status == ("definite" if unevaluated_pass == "first" else "no_data")
    assert served.no_data_reason == (None if unevaluated_pass == "first" else first.no_data_reason)
    assert _states_unevaluated(served, "upstream_error")
    assert served.alternates_left_out


def test_a_capped_alternate_s_refusal_beside_an_evaluated_alternate_is_not_degraded(
    monkeypatch,
) -> None:
    """The tied trade reading's presence is known, so it is evaluated, and materialising
    it refuses the candidate path. The failed read over the widened search's shapes left
    no alternate unevaluated, so the refusal is served as it is, stating nothing and
    cacheable."""
    scenario = _tied_trade_scenario(monkeypatch, ())
    _alternates_presence_read_incomplete(monkeypatch, "upstream_error", _REMIT)
    _seeding_known_trade_presence(monkeypatch)
    refusal = UnsupportedDraft(
        code="scope_too_broad", pointer="/parts/0", detail="the trade reading is capped"
    )
    monkeypatch.setattr(
        candidate_admission, "_materialise_alt_shapes", lambda *args, **kwargs: ([], refusal)
    )

    answer = scenarios._resolve(scenario)

    assert answer.status == "unsupported", answer
    assert not _states_unevaluated(answer, "upstream_error")
    assert not answer.alternates_left_out


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


def test_a_worker_free_after_the_tied_reading_went_unevaluated_makes_no_later_read(
    monkeypatch,
) -> None:
    """The tied reading's read finds no worker beside the elected series' materialisation
    and none at candidate admission. A worker is free after that, and the request sends no
    presence read: the tied reading stays unevaluated and no search widens."""
    scenario = _tied_trade_scenario(monkeypatch, ())
    _alternates_presence_workers_busy(monkeypatch, refusals=2)
    reads = _presence_reads_failing_for(monkeypatch, _TRADE, "no read is made for it")
    reasons = _admission_reasons(monkeypatch)

    answer = scenarios._resolve(scenario)

    assert reads == []
    assert answer.status == "definite", answer
    assert _states_unevaluated(answer, "no_capacity")
    assert reasons == [{1: "no_capacity"}]
    assert "near_neighbor_ambiguous" in [w.code for w in answer.warnings]


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
    """Re-election offers the estimated and the projected count beside the elected one and
    a flagged alternate. Candidate admission takes both siblings as re-election
    materialised them and materialises the flagged alternate alone, so each offered shape
    is materialised once and each sibling's facets are read once."""
    scenario_id = "placeless-reelect-menu-beside-flagged-alternate-together"
    materialised: list[str] = []
    real_materialise = candidates.materialise

    def _materialise(shape, bindings, entities, **kwargs):
        materialised.append(next(iter(shape.sv_arc_facts)))
        return real_materialise(shape, bindings, entities, **kwargs)

    monkeypatch.setattr(candidates, "materialise", _materialise)

    spans = _request_spans(monkeypatch, scenario_id, search_settings)

    assert materialised == [scenarios._MORT_EST, scenarios._MORT_PROJ, scenarios._FAR]
    facet_reads = [
        stat_vars for name, stat_vars, _, _ in spans.spans if name == "observation_facets_batch"
    ]
    for stat_var in (scenarios._MORT_EST, scenarios._MORT_PROJ):
        assert facet_reads.count([stat_var]) == 1, facet_reads


def test_a_role_bearing_elected_shape_reads_no_presence_for_its_siblings(
    monkeypatch, search_settings
) -> None:
    """A role-bearing elected shape on a query naming no place discovers its scope, so
    re-election does not follow it: no presence read and no listing of Earth's countries
    is sent for its subject-only siblings, with or without data on the elected shape."""
    for scenario_id in (
        "placeless-role-bearing-elected-subject-siblings",
        "placeless-role-bearing-elected-with-data-subject-siblings",
    ):
        with monkeypatch.context() as patch:
            spans = _request_spans(patch, scenario_id, search_settings)
        names = {name for name, _, _, _ in spans.spans}
        assert not names & {
            "availability_values",
            "entities_with_data_by_variable",
            "child_places_batch",
        }, names


def test_the_widened_search_s_shapes_ride_the_election_s_presence_read(
    monkeypatch, search_settings
) -> None:
    """When a tie fired, the elected count's scope read covers the tied reading and the
    unrelated count the widened search would add, so the search reads no availability of
    its own."""
    reads = _availability_reads(monkeypatch)
    _request_spans(monkeypatch, "placeless-tie-widened-elsewhere", search_settings)

    assert sorted(reads) == sorted([scenarios._MORT, scenarios._LIVESTOCK, scenarios._FAR])


def test_a_widened_search_that_runs_awaits_its_presence_read_and_probes_over_it(
    monkeypatch, search_settings
) -> None:
    """The tied reading holds no data, so the widened search runs. The widened reading is
    probed over the scope the elected count's scope read gives it, the one country where
    it has data, and no other availability read is sent."""
    scenario = scenarios._SCENARIOS["placeless-tie-widened-elsewhere"]
    scenarios._patch_settings(scenario, search_settings)
    reads = _availability_reads(monkeypatch)
    probed: dict[str, Any] = {}
    real_materialise = candidates.materialise

    def _materialise(shape, bindings, entities, **kwargs):
        probed[next(iter(shape.sv_arc_facts))] = kwargs
        return real_materialise(shape, bindings, entities, **kwargs)

    monkeypatch.setattr(candidates, "materialise", _materialise)

    result = scenarios._resolve(scenario)

    assert result.status == "candidates", result
    assert sorted(reads) == sorted([scenarios._MORT, scenarios._LIVESTOCK, scenarios._FAR])
    widened = probed[scenarios._FAR]
    assert widened["place_dcids"] == [scenarios._TZA]
    assert widened["default_presence"].scope(scenarios._FAR).places == (scenarios._TZA,)


class _Captured(Exception):
    """Ends an option probe once the entities it binds are captured."""


def test_each_option_probe_of_a_lone_candidate_binds_its_own_entities(monkeypatch) -> None:
    """Slot-option enrichment, entity options and the slot-option relaunch of a candidate
    bind the entities of its own place context, not those of the shape in play."""
    own = {"observationAbout": [_GHA, _NOR]}
    spec = SimpleNamespace(
        spec_id="spec_ppp", definition=SimpleNamespace(identity=SimpleNamespace(facets=None))
    )
    grounded = SimpleNamespace(
        cand_places={"spec_ppp": SimpleNamespace(roles=SimpleNamespace(resolved_entities=own))},
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
