"""Fixtures for options-route tests.

The life-expectancy shape is a fixture-only five-tuple absent from the packaged index, so
population comes from graph arcs. The GDP shape carries currency facets for elected-unit
convertibility tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from materialise.graph import Facet as MaterialiseFacet
from materialise.graph import SubjectPage
from search.models import (
    BaseIdentity,
    IdentityBindingSet,
    IdentityBindingUnbound,
    IdentityShapeSchema,
    IdentitySlot,
    IdentityWindowAll,
    OptionsFor,
    OptionsRequest,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, make_resolver
from tests.materialise.conftest import FakeGraphReader

_LIFE_EXPECTANCY_SHAPE = IdentityShapeSchema(
    population_type="Person", measured_property="lifeExpectancy", stat_type="measuredValue"
)
_LIFE_EXPECTANCY_SV = "LifeExpectancy_Person"
_LIFE_EXPECTANCY_ARCS: dict[str, dict[str, tuple[str, ...]]] = {
    _LIFE_EXPECTANCY_SV: {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("lifeExpectancy",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": (),
        "observationProperties": (),
    },
}
_LIFE_EXPECTANCY_SUBJECTS = {
    ("lifeExpectancy", "measuredProperty"): SubjectPage(
        dcids=(_LIFE_EXPECTANCY_SV,), truncated=False
    )
}

_KENYA_DCID = "country/KEN"
_UGANDA_DCID = "country/UGA"

_PLACE_NODES = {
    _KENYA_DCID: {"label": "Kenya", "type": "Country"},
    _UGANDA_DCID: {"label": "Uganda", "type": "Country"},
}
_PLACE_CHILDREN = {"Earth": {"Country": [_KENYA_DCID, _UGANDA_DCID]}}
# Kenya alone has data, so union options must exclude Uganda.
_PLACE_OBS = {
    f"{_LIFE_EXPECTANCY_SV}|{_KENYA_DCID}": [
        {"earliestDate": "1960", "latestDate": "2024", "obsCount": 65},
    ],
}
_PLACE_FACETS = {
    (_LIFE_EXPECTANCY_SV, _KENYA_DCID): [
        MaterialiseFacet(
            facet_id="f1",
            provenance_id="dc/base/TestProvenance",
            measurement_method=None,
            unit=None,
            obs_count=65,
            earliest_date="1960",
            latest_date="2024",
        )
    ],
}


def _place_graph() -> FakeGraph:
    return FakeGraph(
        nodes=_PLACE_NODES,
        obs=_PLACE_OBS,
        detect={},
        resolve={},
        children=_PLACE_CHILDREN,
        enum_members={},
    )


_ACTIVITY_SOURCE_PROP = "activitySource"

# This shape has no packaged-index row, so population comes from graph arcs.
GDP_SHAPE: dict = {
    "kind": "statvar_schema",
    "population_type": "EconomicActivity",
    "measured_property": "amount",
    "stat_type": "measuredValue",
    "measurement_qualifier": "Nominal",
}

_STANDARD_GDP_SV = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"


def _gdp_sv_node() -> dict:
    fixture_path = Path(__file__).parent.parent / "fixtures" / "graph_nodes.json"
    with open(fixture_path) as f:
        return json.load(f)[_STANDARD_GDP_SV]


# Guatemala and the United States are the place options. The GDP node supplies
# the source copied by the second-StatVar fixture.
_GDP_PLACE_NODES = {
    _STANDARD_GDP_SV: _gdp_sv_node(),
    "country/GTM": {"label": "Guatemala", "type": "Country"},
    "country/USA": {"label": "United States", "type": "Country"},
}

_GDP_PLACE_CHILDREN = {
    "Earth": {"Country": ["country/GTM", "country/USA"]},
}


def _standard_place_identity(
    entities: dict[str, list[str]] | None = None,
    *,
    activity_source_refs: list[str] | None = None,
) -> BaseIdentity:
    # The set does not create the extra members; it keeps the identity grammar valid when
    # tests narrow a multi-member family to selected activitySource values.
    binding = (
        IdentityBindingSet(refs=activity_source_refs)
        if activity_source_refs is not None
        else IdentityBindingUnbound()
    )
    return BaseIdentity(
        shape=GDP_SHAPE,
        slots=[IdentitySlot(property=_ACTIVITY_SOURCE_PROP, binding=binding)],
        entities=entities or {},
        requested_window=IdentityWindowAll(),
    )


def _standard_place_request(
    *,
    entities: dict[str, list[str]] | None = None,
    scope: str | None = None,
    activity_source_refs: list[str] | None = None,
    include_coverage: bool = False,
) -> OptionsRequest:
    return OptionsRequest(
        identity=_standard_place_identity(entities, activity_source_refs=activity_source_refs),
        **{"for": OptionsFor(role="observationAbout", scope=scope)},
        include_coverage=include_coverage,
    )


def _place_materialise_graph() -> FakeGraphReader:
    """Serve member arcs and coverage facets through one graph double."""
    return FakeGraphReader(
        arcs=_LIFE_EXPECTANCY_ARCS, subjects=_LIFE_EXPECTANCY_SUBJECTS, facets=_PLACE_FACETS
    )


def _options_identity(
    *,
    entities: dict[str, list[str]] | None = None,
    requested_window: IdentityWindowAll | None = None,
) -> BaseIdentity:
    return BaseIdentity(
        shape=_LIFE_EXPECTANCY_SHAPE,
        slots=[],
        entities=entities or {},
        requested_window=requested_window or IdentityWindowAll(),
    )


def _options_request(
    *,
    entities: dict[str, list[str]] | None = None,
    scope: str | None = None,
    include_coverage: bool = False,
    role: str = "observationAbout",
) -> OptionsRequest:
    return OptionsRequest(
        identity=_options_identity(entities=entities),
        **{"for": OptionsFor(role=role, scope=scope)},
        include_coverage=include_coverage,
    )


def _pin_arm_options_request() -> OptionsRequest:
    """Build a pinned identity rejected before population derivation."""
    identity = BaseIdentity(
        shape={"kind": "stat_var", "id": _LIFE_EXPECTANCY_SV},
        slots=[],
        entities={},
        requested_window=IdentityWindowAll(),
    )
    return OptionsRequest(identity=identity, **{"for": OptionsFor(role="observationAbout")})


def _options_client(
    *,
    sdmx: FakeSdmxClient | None = None,
    materialise_graph: FakeGraphReader | None = None,
) -> TestClient:
    graph = _place_graph()
    app = search_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=sdmx if sdmx is not None else FakeSdmxClient(observations={}),
        materialise_graph=(
            materialise_graph if materialise_graph is not None else _place_materialise_graph()
        ),
    )
    return TestClient(app, raise_server_exceptions=False)
