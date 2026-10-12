"""Validates live responses for all four statuses against the published `/schema`."""

from __future__ import annotations

from typing import get_args

import jsonschema
import pytest
from fastapi.testclient import TestClient

from search.engine.assemble import build_unsupported_part
from search.engine.envelope import make_diagnostics, make_query_echo
from search.engine.spec_build import build_unsupported_definition
from search.models import NoData, NoDataReason, Resolution, Slot
from tests.engine._base_identity import GDP_SHAPE, make_identity, subject_entity
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import member_walk_reader
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, make_resolver


def _client(*, walk: bool = False) -> TestClient:
    """A Search app over the shared fixture graph; with `walk`, a resubmit walks the mint's
    member enumeration over its nodes."""
    graph = FakeGraph()
    app = search_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(),
        materialise_graph=member_walk_reader(graph) if walk else None,
    )
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture(scope="module")
def response_schema() -> dict:
    resp = _client().get("/api/search/schema")
    assert resp.status_code == 200
    return resp.json()["response"]


@pytest.fixture(scope="module")
def resolution_schema(response_schema: dict) -> dict:
    return response_schema["$defs"]["Resolution"]


def _resolve_raw_text(client: TestClient, query: str) -> dict:
    resp = client.post(
        "/api/search/resolve",
        json={"input": {"kind": "raw_text", "query": query}},
    )
    assert resp.status_code == 200
    return resp.json()


def test_definite_response_validates_against_schema(response_schema):
    body = _resolve_raw_text(_client(), "fertility rate in Kenya")
    assert body["status"] == "definite"
    jsonschema.validate(instance=body, schema=response_schema)


def test_response_schema_requires_computed_status(response_schema, resolution_schema):
    body = _resolve_raw_text(_client(), "fertility rate in Kenya")
    assert "status" in resolution_schema["required"]
    body_without_status = {key: value for key, value in body.items() if key != "status"}

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=body_without_status, schema=response_schema)


def test_definite_response_with_facet_options_validates_against_schema(response_schema):
    # Spec.resolved.facet_options only populates under include_option_coverage
    # See FacetOptions' docstring for this option's contract.
    # The default [] alone does not exercise the item schema.
    resp = _client().post(
        "/api/search/resolve",
        json={
            "input": {
                "kind": "raw_text",
                "query": "life expectancy in Kenya",
            },
            "response_options": {"include_option_coverage": True},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "definite"
    facet_options = body["parts"][0]["spec"]["resolved"]["facet_options"]
    assert facet_options
    jsonschema.validate(instance=body, schema=response_schema)


def test_candidates_response_validates_against_schema(response_schema):
    body = _resolve_raw_text(_client(), "under-5 child mortality rate")
    assert body["status"] == "candidates"
    jsonschema.validate(instance=body, schema=response_schema)


def test_no_data_response_validates_against_schema(response_schema):
    body = _resolve_raw_text(_client(), "   ")
    assert body["status"] == "no_data"
    jsonschema.validate(instance=body, schema=response_schema)


def test_no_data_entity_terms_schema_is_additive_to_frozen_reason_enum(response_schema):
    """The structured entity terms are an optional response addition, not a
    new no_data reason or another code channel."""
    no_data_schema = response_schema["$defs"]["NoData"]
    entity_terms_schema = no_data_schema["properties"]["entity_terms"]

    assert entity_terms_schema["type"] == "array"
    assert entity_terms_schema["items"] == {"type": "string"}
    assert "entity_terms" not in no_data_schema["required"]
    assert no_data_schema["properties"]["reason"]["enum"] == list(get_args(NoDataReason))
    assert NoData(reason="entity_not_resolved").model_dump(mode="json")["entity_terms"] == []


def test_unsupported_response_validates_against_schema(response_schema, monkeypatch):
    """The route has no raw-text trigger for `unsupported`, so the test stubs resolution and checks its wire schema."""

    async def _fake_resolve_async(
        request,
        *,
        graph=None,
        resolver=None,
        admission=None,
        llm=None,
        sdmx=None,
        materialise_graph=None,
        materialise_sdmx=None,
        rate_cache=None,
        population_memo=None,
        place_options_pool=None,
        shared_skills_index=None,
        response_cache=None,
        part_memo=None,
        generation=None,
        health=None,
        caller=None,
        budget=None,
    ):
        echo = make_query_echo("price of X in PPP terms", extract_skipped=True)
        diag = make_diagnostics("test-build", [], {}, 0)
        shape = {"kind": "statvar_schema", "population_type": "sh1"}
        definition = build_unsupported_definition(shape, [], entities=None, requested_window=None)
        part = build_unsupported_part(
            "unit_request_unsupported",
            "/identity/facets/unit",
            "a PPP-terms unit request is not expressible at this schema version",
            definition,
            variable_text="price of X in PPP terms",
            warnings=[],
        )
        return Resolution(query_echo=echo, diagnostics=diag, parts=[part])

    monkeypatch.setattr("search.engine.app.resolve_async", _fake_resolve_async)
    client = _client()
    resp = client.post(
        "/api/search/resolve",
        json={"input": {"kind": "raw_text", "query": "price of X in PPP terms"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "unsupported"
    jsonschema.validate(instance=body, schema=response_schema)


def test_observation_series_definition_pins_role_tuple_shape(response_schema):
    definition = response_schema["$defs"]["ObservationSeries"]
    assert set(definition["required"]) >= {"stat_var", "entities", "points", "total_obs"}
    properties = definition["properties"]
    assert "stat_var_dcid" not in properties
    assert "entity_dcid" not in properties


def test_resubmit_door_response_validates_against_schema(response_schema):
    client = _client(walk=True)
    entities = [subject_entity("country/IND")]
    resp = client.post(
        "/api/search/resolve",
        json={
            "input": {
                "kind": "dataset_resubmit",
                "entries": [
                    {
                        "entry_kind": "identity",
                        "variable_text": "a resubmitted identity",
                        "identity": make_identity(
                            GDP_SHAPE,
                            [
                                Slot.model_validate(
                                    {
                                        "key": {
                                            "axis": "how",
                                            "property": {
                                                "dcid": "activitySource",
                                                "label": "activity source",
                                            },
                                            "label": "activity source",
                                        },
                                        "binding": {"kind": "unbound"},
                                    }
                                )
                            ],
                            entities=entities,
                        ).model_dump(mode="json"),
                    }
                ],
            }
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "definite"
    jsonschema.validate(instance=body, schema=response_schema)


def test_resolved_scope_schema_describes_place_and_directional_defaults(response_schema):
    """The public schema must cover every supported defaulted entity axis.

    Scope is per-Spec while diagnostics are response-level, so the text must
    name both warning codes without claiming an exact warning-to-Spec mapping.
    """
    scope_description = response_schema["$defs"]["ResolvedScope"]["description"]
    resolved_scope_description = response_schema["$defs"]["ResolvedSpec"]["properties"]["scope"][
        "description"
    ]

    for description in (scope_description, resolved_scope_description):
        for token in (
            "observationAbout",
            "origin",
            "destination",
            "place_scope_defaulted",
            "role_scope_defaulted",
        ):
            assert token in description
        assert "one-to-one" in description or "not an exact per-Spec" in description


def test_available_at_place_schema_describes_lifetime_availability(response_schema):
    description = response_schema["$defs"]["ResolvedSpec"]["properties"]["available_at_place"][
        "description"
    ]

    assert "lifetime availability at the resolved place" in description
    assert (
        "CoverageExact.has_data may be False for a requested window even when this field is True"
        in description
    )


def test_rendered_sentence_schema_describes_axis_routing_not_parentheses(resolution_schema):
    """The schema documents hoisting over definite parts and the per-part fallback.

    `origin` is identified by role. A differing comma-connective qualifier returns one
    sentence per part; no attribution clause is written.
    """
    description = resolution_schema["properties"]["rendered_sentence"]["description"]

    assert "When every contributing part is definite" in description
    assert "role=='origin'" in description
    assert "SentenceAxis has no 'origin' member" in description
    assert "A qualifier that differs stays inline in each part's own phrase" in description
    assert "connective opens with a comma" in description
    assert "attribution" not in description
    assert "parenthes" not in description.lower()


def test_sentence_plan_schema_names_the_fallback_glue(resolution_schema):
    description = resolution_schema["properties"]["sentence_plan"]["description"]

    assert "the shared ' ' the per-part fallback puts between two sentences" in description
    assert "attribution" not in description


def test_sentence_token_part_index_schema_states_which_tokens_are_shared(response_schema):
    """The schema distinguishes indexed variable and inline qualifier tokens from hoisted
    tokens. `origin` is identified by role."""
    description = response_schema["$defs"]["SentenceToken"]["properties"]["part_index"][
        "description"
    ]

    assert "a variable phrase token, or a qualifier attached to it" in description
    assert "where, when, or source" in description
    assert "role=='origin'" in description
    assert "attribution" not in description
    assert "note" not in description
    assert "parenthes" not in description


def test_sentence_token_kind_schema_offers_text_and_slot_only(response_schema):
    kind = response_schema["$defs"]["SentenceToken"]["properties"]["kind"]

    assert kind["enum"] == ["text", "slot"]
    assert "attribution" not in kind["description"]


def _iter_descriptions(node: object):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "description" and isinstance(value, str):
                yield value
            else:
                yield from _iter_descriptions(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_descriptions(item)


def test_no_served_schema_description_mentions_a_parenthesis():
    """Served schema descriptions must not claim the composer mints parentheses."""
    resp = _client().get("/api/search/schema")
    assert resp.status_code == 200
    document = resp.json()

    for description in _iter_descriptions(document):
        assert "parenthes" not in description.lower()
