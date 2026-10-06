"""Every emitted definition hashes to its ``spec_id`` on definite, candidate, and resubmit paths.

Unchanged definitions verify and mint the same id when reposted."""

from __future__ import annotations

import hashlib

import pytest
import rfc8785
from dfo_spec_kit import identity
from fastapi.testclient import TestClient

from search.engine.region import resolve_spec_resubmit
from search.models import (
    CandidatesPart,
    ResubmitIdentityPayload,
)
from tests.engine._base_identity import (
    make_identity,
    subject_entity,
)
from tests.engine._builders import value_slot as _value_slot
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    make_request,
    member_walk_reader,
    offline_resolve,
    walk_options,
)
from tests.engine._options_fixtures import _gdp_sv_node
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, _load_json, make_resolver

_GDP_SV_DCID = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
# Resubmit derives this five-tuple from the fixture StatVar's graph node.
_GDP_SHAPE = {
    "kind": "statvar_schema",
    "population_type": "EconomicActivity",
    "measured_property": "amount",
    "stat_type": "measuredValue",
    "measurement_qualifier": "Nominal",
}
_GDP_ENTITY_DCID = "country/IND"

# The base and posted identities both declare this activity-source slot.
_ACTIVITY_SOURCE_SLOT = _value_slot(
    "how",
    "activitySource",
    "activity source",
    "GrossDomesticProduction",
    "Gross Domestic Production",
)


def _assert_definition_hashes_to_spec_id(spec) -> None:
    """The wire identity hashes to ``spec_id``.

    Both copies of ``spec_id`` and schema version 1 agree."""
    dumped = spec.definition.identity.model_dump(exclude_none=True)
    assert identity.spec_id_from_value(dumped) == spec.spec_id
    assert spec.definition.spec_id == spec.spec_id
    assert spec.definition.schema_version == 1


# ---------------------------------------------------------------------------
# Primary spec: raw_text, place_as_constraint ON (the common case)
# ---------------------------------------------------------------------------


def test_primary_definite_spec_definition_hashes_to_spec_id():
    result = offline_resolve(
        make_request("fertility rate in Kenya"), pac=True, sdmx=FakeSdmxClient()
    ).parts[0]
    assert result.status == "definite"
    _assert_definition_hashes_to_spec_id(result.spec)


# ---------------------------------------------------------------------------
# place_as_constraint=False mode
# ---------------------------------------------------------------------------


def test_place_constraint_off_spec_definition_hashes_to_spec_id():
    """With ``pac=False``, the emitted definition hashes to ``spec_id``."""
    result = offline_resolve(
        make_request("fertility rate in Kenya"), pac=False, sdmx=FakeSdmxClient()
    ).parts[0]
    assert result.status == "definite"
    spec = result.spec
    assert all(e.role == "observationAbout" for e in spec.resolved.entities)
    _assert_definition_hashes_to_spec_id(spec)


# ---------------------------------------------------------------------------
# Candidates response: no single answer resolved
# ---------------------------------------------------------------------------


def test_candidates_response_specs_definitions_hash_to_their_spec_ids():
    """Every candidate definition hashes to its own ``spec_id``."""
    result = offline_resolve(make_request("under-5 child mortality rate")).parts[0]
    assert isinstance(result, CandidatesPart)
    assert len(result.candidates.specs) >= 2
    for spec in result.candidates.specs:
        _assert_definition_hashes_to_spec_id(spec)


# ---------------------------------------------------------------------------
# The served wire body, hashed as a consumer hashes it
# ---------------------------------------------------------------------------

_G7_DCIDS = (
    "country/CAN",
    "country/DEU",
    "country/FRA",
    "country/GBR",
    "country/ITA",
    "country/JPN",
    "country/USA",
)
_GDP_OBS_ROW = {
    "earliestDate": "2020",
    "latestDate": "2020",
    "obsCount": 1,
    "provenanceId": "dc/base/PlaceFixture",
}


_G7_NODE = "GroupOfSeven"


def _group_graph(*, g7_node: bool = False) -> FakeGraph:
    """Holds the G7 countries and the European Union's own node, and no node for the G7.
    With `g7_node` it also holds `_G7_NODE`, which contains each G7 country."""
    contained = {"arcs": {"containedInPlace": {"nodes": [{"dcid": _G7_NODE}]}}} if g7_node else {}
    return FakeGraph(
        nodes={
            _GDP_SV_DCID: _gdp_sv_node(),
            **{dcid: {"label": dcid, "type": "Country", **contained} for dcid in _G7_DCIDS},
            "EuropeanUnion": {"label": "European Union", "type": "Place"},
            **({_G7_NODE: {"label": "Group of Seven"}} if g7_node else {}),
        },
        obs={f"{_GDP_SV_DCID}|{dcid}": [_GDP_OBS_ROW] for dcid in _G7_DCIDS},
        detect={"gdp": {"svs": [_GDP_SV_DCID], "cosine_scores": [0.95]}},
        resolve={},
        places={},
        children={},
    )


# Count_Person and the members of its smallest recorded group, dc/g/Demographics.
_DEMOGRAPHICS_SVS = (
    "Count_Person",
    "GrowthRate_Count_Person",
    "LifeExpectancy_Person",
    "Count_Person_PerArea",
)


def _related_measures_graph() -> FakeGraph:
    """The recorded graph, with Kenya's population facets and those of its related measures
    at every G7 country, so a G7 population answer serves related measures."""
    recorded = _load_json("graph_obs.json")
    return FakeGraph(
        nodes={
            **_load_json("graph_nodes.json"),
            **{dcid: {"label": dcid, "type": "Country"} for dcid in _G7_DCIDS},
        },
        obs={
            **recorded,
            **{
                f"{stat_var}|{dcid}": recorded[f"{stat_var}|country/KEN"]
                for stat_var in _DEMOGRAPHICS_SVS
                for dcid in _G7_DCIDS
            },
        },
    )


def _served_definitions(node: object) -> list[dict]:
    """Every JSON object under `node` that carries both `spec_id` and `identity`."""
    if isinstance(node, list):
        return [found for item in node for found in _served_definitions(item)]
    if not isinstance(node, dict):
        return []
    own = [node] if "spec_id" in node and "identity" in node else []
    return own + [found for value in node.values() for found in _served_definitions(value)]


@pytest.mark.parametrize(
    ("build_graph", "variable_text", "entity_text", "served_group", "derived"),
    [
        (_group_graph, "gdp", "Canada", None, False),
        (
            _group_graph,
            "gdp",
            "European Union",
            {"dcid": "EuropeanUnion", "label": "European Union"},
            False,
        ),
        (_group_graph, "gdp", "G7", {"label": "G7"}, False),
        (_related_measures_graph, "population", "G7", {"label": "G7"}, True),
    ],
    ids=[
        "no_group",
        "group_with_graph_node",
        "group_without_graph_node",
        "group_with_related_measures",
    ],
)
def test_served_identity_hashes_whole_to_its_spec_id(
    build_graph, variable_text, entity_text, served_group, derived
):
    """A served ``identity`` is the JCS document its ``spec_id`` hashes.

    The check uses ``rfc8785`` and ``hashlib`` on the response bytes alone. A group the request
    names is served at ``resolved.entity_groups``. A ``derived`` row also serves definitions
    Search derives from the answer's own, here its related measures."""
    graph = build_graph()
    app = search_app(
        graph=graph, resolver=make_resolver(graph), llm=FakeLLM(), sdmx=FakeSdmxClient()
    )
    response = TestClient(app).post(
        "/api/search/resolve",
        json={
            "input": {
                "kind": "parsed",
                "parts": [{"variable_text": variable_text, "entities": [{"text": entity_text}]}],
            }
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    part = body["parts"][0]
    assert part["status"] == "definite", part
    served = part["spec"]["resolved"].get("entity_groups")
    assert served == (served_group and {"observationAbout": served_group})

    definitions = _served_definitions(body)
    assert definitions
    if derived:
        assert len(definitions) >= 2
        assert {d["spec_id"] for d in definitions} - {part["spec"]["spec_id"]}
    for definition in definitions:
        computed_from_the_whole_identity = (
            "spec_" + hashlib.sha256(rfc8785.dumps(definition["identity"])).hexdigest()[:32]
        )
        assert computed_from_the_whole_identity == definition["spec_id"]


def _g7_first_answer_and_client() -> tuple[TestClient, dict]:
    """A client over a graph that holds a G7 node, and the Spec it serves for G7 GDP."""
    graph = _group_graph(g7_node=True)
    app = search_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(),
        materialise_graph=member_walk_reader(graph),
    )
    client = TestClient(app)
    response = client.post(
        "/api/search/resolve",
        json={
            "input": {
                "kind": "parsed",
                "parts": [{"variable_text": "gdp", "entities": [{"text": "G7"}]}],
            }
        },
    )
    assert response.status_code == 200, response.text
    return client, response.json()["parts"][0]["spec"]


def _resubmit(client: TestClient, entry: dict):
    return client.post(
        "/api/search/resolve",
        json={
            "input": {
                "kind": "dataset_resubmit",
                "entries": [{"entry_kind": "identity", "variable_text": "gdp", **entry}],
            }
        },
    )


def test_resubmitted_identity_with_a_group_beside_it_hashes_whole_to_its_spec_id():
    """A served identity posted back with a graph-confirmed group beside it serves the same
    ``spec_id``, every served ``identity`` is the JCS document its id hashes, and the group
    is served at ``resolved.entity_groups`` under the graph's label."""
    client, first = _g7_first_answer_and_client()

    response = _resubmit(
        client,
        {
            "identity": first["definition"]["identity"],
            "entity_groups": {"observationAbout": {"dcid": _G7_NODE, "label": "G7"}},
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    part = body["parts"][0]
    assert part["status"] == "definite", part
    assert part["spec"]["spec_id"] == first["spec_id"]
    assert part["spec"]["resolved"]["entity_groups"] == {
        "observationAbout": {"dcid": _G7_NODE, "label": "Group of Seven"}
    }
    definitions = _served_definitions(body)
    assert definitions
    for definition in definitions:
        assert set(definition["identity"]["entities"]["observationAbout"]) == set(_G7_DCIDS)
        whole = "spec_" + hashlib.sha256(rfc8785.dumps(definition["identity"])).hexdigest()[:32]
        assert whole == definition["spec_id"]


def test_resubmit_refuses_an_identity_that_carries_entity_groups():
    """``entity_groups`` belongs beside ``identity``. Posted inside ``identity`` it is refused as
    ``request_invalid``, and the same group posted beside the identity is accepted."""
    client, first = _g7_first_answer_and_client()
    identity_document = first["definition"]["identity"]
    groups = {"observationAbout": {"label": "G7"}}

    accepted = _resubmit(client, {"identity": identity_document, "entity_groups": groups})
    refused = _resubmit(client, {"identity": {**identity_document, "entity_groups": groups}})

    assert accepted.status_code == 200, accepted.text
    assert refused.status_code == 400
    problem = refused.json()
    assert problem["code"] == "request_invalid"
    assert problem["pointer"] == "/input/entries/0/identity/entity_groups"


# ---------------------------------------------------------------------------
# the resubmit door: standard promote
# ---------------------------------------------------------------------------


def test_standard_resubmit_definition_hashes_to_spec_id():
    entities = [subject_entity(_GDP_ENTITY_DCID)]
    inp = ResubmitIdentityPayload(
        identity=make_identity(_GDP_SHAPE, [_ACTIVITY_SOURCE_SLOT], entities=entities),
    )
    region = resolve_spec_resubmit(
        inp=inp,
        graph=FakeGraph(),
        sdmx=FakeSdmxClient(),
        coverage=empty_coverage_snapshot(),
        options=walk_options(FakeGraph()),
    )
    assert region.status == "definite"
    _assert_definition_hashes_to_spec_id(region.specs[0])


# ---------------------------------------------------------------------------
# Round-trip: an emitted definition, reposted unchanged, verifies and mints
# the same spec_id
# ---------------------------------------------------------------------------


def test_emitted_definition_hashes_to_its_own_spec_id():
    """An emitted candidate definition hashes to its own ``spec_id``."""
    result = offline_resolve(make_request("under-5 child mortality rate")).parts[0]
    assert isinstance(result, CandidatesPart)
    _assert_definition_hashes_to_spec_id(result.candidates.specs[0])


def test_reposting_emitted_definition_unchanged_mints_the_same_spec_id():
    """Reposting an emitted definition mints the same ``spec_id``."""
    entities = [subject_entity(_GDP_ENTITY_DCID)]
    identity = make_identity(_GDP_SHAPE, [_ACTIVITY_SOURCE_SLOT], entities=entities)
    inp = ResubmitIdentityPayload(
        identity=identity,
    )
    first = resolve_spec_resubmit(
        inp=inp,
        graph=FakeGraph(),
        sdmx=FakeSdmxClient(),
        coverage=empty_coverage_snapshot(),
        options=walk_options(FakeGraph()),
    )
    assert first.status == "definite"
    first_spec = first.specs[0]

    # Re-post the FIRST response's own emitted definition, completely
    # unedited, as the base for a second resubmit of the identical slots.
    second_base_identity = first_spec.definition.identity
    inp2 = ResubmitIdentityPayload(
        identity=second_base_identity,
    )
    second = resolve_spec_resubmit(
        inp=inp2,
        graph=FakeGraph(),
        sdmx=FakeSdmxClient(),
        coverage=empty_coverage_snapshot(),
        options=walk_options(FakeGraph()),
    )
    assert second.status == "definite"
    assert second.specs[0].spec_id == first_spec.spec_id
