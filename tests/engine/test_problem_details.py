"""Check RFC 9457 problem details, refusal codes, and trace-ID consistency."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from search.engine.config import settings
from tests.engine._base_identity import (
    GDP_SHAPE,
    make_identity,
    subject_entity,
)
from tests.engine._builders import unbound_slot as _unbound_slot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import PINNED_DATE, member_walk_reader
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, make_resolver


def make_client() -> TestClient:
    graph = FakeGraph()
    app = search_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        materialise_graph=member_walk_reader(graph),
    )
    return TestClient(app, raise_server_exceptions=False)


def _identity(
    shape: dict,
    *,
    slots: list[dict] | None = None,
    entities: dict[str, list[str]] | None = None,
    facets: dict | None = None,
) -> dict:
    identity = {
        "shape": shape,
        "slots": slots or [],
        "entities": entities or {},
        "requested_window": {"kind": "all"},
    }
    if facets is not None:
        identity["facets"] = facets
    return identity


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into [(event_type, data_dict), ...], skipping `: ping` comments."""
    frames: list[tuple[str, dict]] = []
    event_type: str | None = None
    for line in body.splitlines():
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:") and event_type is not None:
            frames.append((event_type, json.loads(line[len("data:") :].strip())))
            event_type = None
    return frames


def _assert_problem_shape(resp, *, status: int, code: str, retryable: bool) -> dict:
    assert resp.status_code == status
    assert resp.headers["content-type"].startswith("application/problem+json")
    # 501 is heuristically cacheable, so refusals use `no-store`.
    assert resp.headers["cache-control"] == "no-store"
    body = resp.json()
    assert body["type"] == f"/codes/{code}"
    assert body["status"] == status
    assert body["code"] == code
    assert body["retryable"] is retryable
    assert body["capability_version"] == settings().capability_version
    assert isinstance(body["trace_id"], str) and body["trace_id"]
    assert body.get("title")
    assert body.get("detail")
    return body


class TestRequestInvalid:
    def test_missing_required_field_is_request_invalid_400(self):
        resp = make_client().post(
            "/api/search/resolve",
            json={"not_valid": "body"},
        )
        body = _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)
        # The missing field has nothing to "resolve" against (the posted
        # body doesn't have it), so the pointer names the parent /input
        # rather than coming back empty.
        assert body["pointer"] == "/input"

    def test_unknown_input_kind_is_request_invalid_400(self):
        resp = make_client().post(
            "/api/search/resolve",
            json={"input": {"kind": "not_a_real_kind"}},
        )
        _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)

    def test_retired_spec_resubmit_kind_is_refused_and_dataset_resubmit_serves(self):
        """`spec_resubmit` is rejected; the same identity works as a dataset-resubmit entry."""
        entities = [subject_entity("country/IND")]
        identity = make_identity(
            GDP_SHAPE,
            [_unbound_slot("how", "activitySource", "activity source")],
            entities=entities,
        ).model_dump(mode="json")
        client = make_client()
        refused = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "spec_resubmit",
                    "identity": identity,
                }
            },
        )
        body = _assert_problem_shape(refused, status=400, code="request_invalid", retryable=False)
        assert "spec_resubmit" in body["detail"]
        assert body["pointer"] == "/input"

        served = client.post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "nominal GDP of India",
                            "identity": identity,
                        }
                    ],
                }
            },
        )
        assert served.status_code == 200, served.text
        assert served.json()["status"] == "definite"

    def test_pointer_derived_from_first_validation_error(self):
        resp = make_client().post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": 12345}},
        )
        body = _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)
        assert body["pointer"] is not None
        assert "query" in body["pointer"]

    def test_unknown_top_level_field_is_request_invalid_400(self):
        # The closed schema returns Pydantic's extra-field message and pointer.
        resp = make_client().post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "x"}, "bogus_field": True},
        )
        body = _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)
        assert body["pointer"] == "/bogus_field"
        assert body["detail"] == "Extra inputs are not permitted"

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("stat_var_dcids", ["Amount_EconomicActivity_GrossDomesticProduction_Nominal"]),
            ("entity_dcids", ["country/IND"]),
        ],
    )
    def test_a_removed_promote_field_is_refused_as_unknown(self, field: str, value: list[str]):
        """A removed promote field refuses like any unknown field, with the platform detail."""
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": _identity(GDP_SHAPE),
                            field: value,
                        }
                    ],
                }
            },
        )
        body = _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)
        assert body["pointer"] == f"/input/entries/0/{field}"
        assert body["detail"] == "Extra inputs are not permitted"

    @pytest.mark.parametrize("option", ["max_candidates", "include_sentence"])
    def test_a_removed_response_option_is_refused_as_unknown(self, option: str):
        """Removed response options refuse like any unknown field, with the platform detail."""
        value = {"max_candidates": 4, "include_sentence": True}[option]
        entry = {
            "entry_kind": "identity",
            "variable_text": "a resubmitted identity",
            "identity": _identity(GDP_SHAPE),
        }
        request = {
            "input": {"kind": "dataset_resubmit", "entries": [entry]},
            "response_options": {option: value},
        }
        resp = make_client().post("/api/search/resolve", json=request)
        problem = _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)
        assert problem["pointer"] == f"/response_options/{option}"
        assert problem["detail"] == "Extra inputs are not permitted"

    def test_unknown_field_inside_input_is_request_invalid_400(self):
        # The pointer omits `raw_text`, which is a discriminator value, not a key.
        resp = make_client().post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "x", "bogus_field": True}},
        )
        body = _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)
        assert body["pointer"] == "/input/bogus_field"

    def test_malformed_json_body_omits_pointer(self):
        # A byte offset is not a JSON Pointer.
        resp = make_client().post(
            "/api/search/resolve",
            content=b'{"input": {"kind": "raw_text", "query": ',
            headers={"Content-Type": "application/json"},
        )
        body = _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)
        assert "pointer" not in body

    def test_unknown_field_inside_options_is_request_invalid_400(self):
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "input": {"kind": "raw_text", "query": "x"},
                "response_options": {"bogus_option": True},
            },
        )
        _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)


class TestUnsupportedSchemaVersion:
    def test_wrong_schema_version_is_422(self):
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "schema_version": "2.0",
                "input": {"kind": "raw_text", "query": "x"},
            },
        )
        body = _assert_problem_shape(
            resp, status=422, code="unsupported_schema_version", retryable=False
        )
        assert body["pointer"] == "/schema_version"
        assert body["detail"] != "Bad request"
        assert "schema_version" in body["detail"]

    def test_wrong_schema_version_wins_over_a_coexisting_extra_field(self):
        # Schema-version errors take precedence over extra-field errors.
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "schema_version": "2.0",
                "input": {"kind": "raw_text", "query": "x", "bogus_field": True},
            },
        )
        body = _assert_problem_shape(
            resp, status=422, code="unsupported_schema_version", retryable=False
        )
        assert body["pointer"] == "/schema_version"

    def test_boolean_schema_version_is_422(self):
        # Pydantic's lax mode can coerce `true` to integer 1.
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "schema_version": True,
                "input": {"kind": "raw_text", "query": "x"},
            },
        )
        body = _assert_problem_shape(
            resp, status=422, code="unsupported_schema_version", retryable=False
        )
        assert body["pointer"] == "/schema_version"

    def test_float_schema_version_is_422(self):
        # Pydantic's lax mode can coerce `1.0` to integer 1.
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "schema_version": 1.0,
                "input": {"kind": "raw_text", "query": "x"},
            },
        )
        body = _assert_problem_shape(
            resp, status=422, code="unsupported_schema_version", retryable=False
        )
        assert body["pointer"] == "/schema_version"


class TestDoorNotImplemented:
    def test_a_door_named_in_the_unimplemented_set_is_door_not_implemented_501(self):
        # Patch the empty set to exercise the generic refusal for an unimplemented door.
        with patch("search.engine.app.SEARCH_UNIMPLEMENTED_DOORS", frozenset({"raw_text"})):
            resp = make_client().post(
                "/api/search/resolve",
                json={"input": {"kind": "raw_text", "query": "x"}},
            )
        body = _assert_problem_shape(resp, status=501, code="door_not_implemented", retryable=False)
        assert body["pointer"] == "/input/kind"
        assert body["detail"] != "Bad request"
        assert "capabilities" in body["detail"]


class TestQueryTooLong:
    def test_query_too_long_is_422(self):

        resp = make_client().post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "x" * (settings().max_query_chars + 1)}},
        )
        body = _assert_problem_shape(resp, status=422, code="query_too_long", retryable=False)
        assert body["pointer"] == "/input/query"
        assert body["detail"] != "Bad request"
        assert str(settings().max_query_chars) in body["detail"]


class TestResubmitIdentityRefusals:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("base_spec_id", "spec_" + "0" * 32),
            ("base_identity", _identity(GDP_SHAPE)),
            ("removed", {"entities": ["origin"]}),
            (
                "resolved_stat_var_dcids",
                ["Amount_EconomicActivity_GrossDomesticProduction_Nominal"],
            ),
        ],
    )
    def test_a_removed_field_is_refused_at_its_pointer(self, field, value):
        """The identity alone defines a resubmit. A posted base or member list is an unknown
        field, refused before any read with the platform detail."""
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": _identity(GDP_SHAPE),
                            field: value,
                        }
                    ],
                }
            },
        )
        body = _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)
        assert body["pointer"] == f"/input/entries/0/{field}"
        assert body["detail"] == "Extra inputs are not permitted"

    def test_a_candidate_posting_members_is_refused_at_its_pointer(self):
        candidate = {"identity": _identity(GDP_SHAPE)}
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "candidates",
                            "variable_text": "GDP",
                            "ordering": "intent_rank",
                            "truncated": False,
                            "candidates": [
                                {**candidate, "resolved_stat_var_dcids": ["Count_Person"]},
                                candidate,
                            ],
                        }
                    ],
                }
            },
        )
        body = _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)
        assert body["pointer"] == "/input/entries/0/candidates/0/resolved_stat_var_dcids"
        assert body["detail"] == "Extra inputs are not permitted"

    # Malformed identities fail request validation with a 400 response.

    def test_malformed_identity_missing_ref_is_request_invalid_400(self):
        """A value binding requires a `ref`."""
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": {
                                "shape": {
                                    "kind": "statvar_schema",
                                    "population_type": "some_shape",
                                },
                                "slots": [{"property": "someProp", "binding": {"state": "value"}}],
                                "entities": {},
                                "requested_window": {"kind": "all"},
                            },
                        }
                    ],
                }
            },
        )
        _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)

    def test_malformed_identity_missing_property_is_request_invalid_400(self):
        """An identity slot requires a property."""
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": {
                                "shape": {
                                    "kind": "statvar_schema",
                                    "population_type": "some_shape",
                                },
                                "slots": [{"binding": {"state": "unbound"}}],
                                "entities": {},
                                "requested_window": {"kind": "all"},
                            },
                        }
                    ],
                }
            },
        )
        _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)

    def test_malformed_identity_unrecognised_facet_axis_is_request_invalid_400(self):
        """BaseIdentity rejects facet axes outside its closed vocabulary."""
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": {
                                "shape": {
                                    "kind": "statvar_schema",
                                    "population_type": "some_shape",
                                },
                                "slots": [],
                                "entities": {},
                                "requested_window": {"kind": "all"},
                                "facets": {"bogus_axis": {"state": "unbound"}},
                            },
                        }
                    ],
                }
            },
        )
        _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)

    def test_resubmit_facets_unrecognised_axis_is_request_invalid_400(self):
        """Resubmit facets reject unknown axes independently of `BaseIdentity`."""
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": _identity(
                                {"kind": "statvar_schema", "population_type": "some_shape"},
                                facets={"bogus_axis": {"state": "unbound"}},
                            ),
                        }
                    ],
                }
            },
        )
        _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)

    def test_facet_binding_extra_key_pointer_names_the_real_path(self):
        """#666: the facet discriminator must preserve the posted key in the pointer."""
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": _identity(
                                GDP_SHAPE,
                                facets={"provenance": {"state": "value", "ref": "ok", "bogus": 1}},
                            ),
                        }
                    ],
                }
            },
        )
        body = _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)
        assert body["pointer"] == "/input/entries/0/identity/facets/provenance/bogus"

    def test_malformed_identity_empty_entity_role_is_request_invalid_400(self):
        """Each entity role needs at least one DCID before hashing."""
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": {
                                "shape": {
                                    "kind": "statvar_schema",
                                    "population_type": "some_shape",
                                },
                                "slots": [],
                                "entities": {"destination": []},
                                "requested_window": {"kind": "all"},
                            },
                        }
                    ],
                }
            },
        )
        _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)

    def test_malformed_identity_unbound_facet_binding_is_request_invalid_400(self):
        """The sparse facet map omits unbound axes."""
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": {
                                "shape": {
                                    "kind": "statvar_schema",
                                    "population_type": "some_shape",
                                },
                                "slots": [],
                                "entities": {},
                                "requested_window": {"kind": "all"},
                                "facets": {"provenance": {"state": "unbound"}},
                            },
                        }
                    ],
                }
            },
        )
        _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)

    def test_malformed_identity_unbound_facet_binding_pointer_is_exact(self):
        """An unbound facet error points to the facet path without a union tag."""
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": {
                                "shape": {
                                    "kind": "statvar_schema",
                                    "population_type": "some_shape",
                                },
                                "slots": [],
                                "entities": {},
                                "requested_window": {"kind": "all"},
                                "facets": {"provenance": {"state": "unbound"}},
                            },
                        }
                    ],
                }
            },
        )
        body = _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)
        assert body["pointer"] == "/input/entries/0/identity/facets/provenance"

    def test_reversed_requested_window_is_request_invalid_400(self):
        # Shape and StatVar match, so the reversed window determines validation.
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": {
                                "shape": GDP_SHAPE,
                                # activitySource is this SV's only constraint, posted
                                # unbound, so the reversed window is the only fault.
                                "slots": [
                                    {
                                        "key": {
                                            "axis": "how",
                                            "property": {
                                                "ref": "activitySource",
                                                "label": "activity source",
                                            },
                                            "label": "activity source",
                                        },
                                        "binding": {"kind": "unbound"},
                                    },
                                ],
                                "entities": {"observationAbout": ["country/IND"]},
                                "requested_window": {"start_year": 2025, "end_year": 2010},
                            },
                        }
                    ],
                }
            },
        )
        _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)

    def test_source_slot_and_facets_provenance_conflict_is_request_invalid_400(self):
        """ResubmitIdentityPayload's own model validator raises a conflict error
        when a request posts a source-axis slot and facets.provenance
        together, since provenance binds through exactly one of the two
        channels."""
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": _identity(
                                {"kind": "statvar_schema", "population_type": "some_shape"}
                            ),
                            "facets": {"provenance": {"state": "value", "ref": "y"}},
                        }
                    ],
                }
            },
        )
        _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)

    def test_request_form_facet_binding_on_non_unit_axis_is_request_invalid_400(self):
        """BaseIdentity's model validator: the unit request form is only
        legal on the unit axis."""
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": {
                                "shape": {
                                    "kind": "statvar_schema",
                                    "population_type": "some_shape",
                                },
                                "slots": [],
                                "entities": {},
                                "requested_window": {"kind": "all"},
                                "facets": {
                                    "measurementMethod": {
                                        "state": "value",
                                        "request": {"currency": "USD", "prices": "current"},
                                    }
                                },
                            },
                        }
                    ],
                }
            },
        )
        _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)

    def test_request_form_on_resubmit_facets_non_unit_axis_is_request_invalid_400(self):
        """ResubmitIdentityPayload.facets carries the identical rule, enforced by
        its own model validator -- this door receives a directly-posted
        ResolveRequest.inp with no BaseIdentity validation in between, so the
        rule above alone would not catch a request form posted here."""
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": _identity(
                                {"kind": "statvar_schema", "population_type": "some_shape"},
                                facets={
                                    "measurementMethod": {
                                        "state": "value",
                                        "request": {"currency": "USD", "prices": "current"},
                                    }
                                },
                            ),
                        }
                    ],
                }
            },
        )
        _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)

    def test_unit_request_constant_prices_without_base_year_is_request_invalid_400(self):
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": {
                                "shape": {
                                    "kind": "statvar_schema",
                                    "population_type": "some_shape",
                                },
                                "slots": [],
                                "entities": {},
                                "requested_window": {"kind": "all"},
                                "facets": {
                                    "unit": {
                                        "state": "value",
                                        "request": {"currency": "USD", "prices": "constant"},
                                    }
                                },
                            },
                        }
                    ],
                }
            },
        )
        _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)

    def test_unit_request_lowercase_currency_is_request_invalid_400(self):
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": {
                                "shape": {
                                    "kind": "statvar_schema",
                                    "population_type": "some_shape",
                                },
                                "slots": [],
                                "entities": {},
                                "requested_window": {"kind": "all"},
                                "facets": {
                                    "unit": {
                                        "state": "value",
                                        "request": {"currency": "usd", "prices": "current"},
                                    }
                                },
                            },
                        }
                    ],
                }
            },
        )
        _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)


class TestIdentityFacetBindingUnion:
    """IdentityFacetBinding's callable discriminator (models._facet_binding_tag)
    resolves each of its three arms to the intended class -- pure model-level
    coverage, no HTTP involved. See test_malformed_identity_unbound_facet_binding_pointer_is_exact
    (above) for the malformed-input side of this same union.
    """

    def test_ref_form_resolves_to_identity_binding_value(self):
        from pydantic import TypeAdapter

        from search.models import IdentityBindingValue, IdentityFacetBinding

        binding = TypeAdapter(IdentityFacetBinding).validate_python({"state": "value", "ref": "x"})
        assert isinstance(binding, IdentityBindingValue)
        assert binding.ref == "x"

    def test_set_form_resolves_to_identity_binding_set(self):
        from pydantic import TypeAdapter

        from search.models import IdentityBindingSet, IdentityFacetBinding

        binding = TypeAdapter(IdentityFacetBinding).validate_python(
            {"state": "set", "refs": ["a", "b"]}
        )
        assert isinstance(binding, IdentityBindingSet)
        assert binding.refs == ["a", "b"]

    def test_request_form_resolves_to_identity_facet_value_request(self):
        from pydantic import TypeAdapter

        from search.models import IdentityFacetBinding, IdentityFacetValueRequest

        binding = TypeAdapter(IdentityFacetBinding).validate_python(
            {"state": "value", "request": {"currency": "USD", "prices": "current"}}
        )
        assert isinstance(binding, IdentityFacetValueRequest)
        assert binding.request.currency == "USD"

    def test_both_ref_and_request_present_reports_union_tag_not_found(self):
        """Both `ref` and `request` present reports the same tag-not-found
        shape a completely unrecognised binding gets."""
        from pydantic import TypeAdapter, ValidationError

        from search.models import IdentityFacetBinding

        with pytest.raises(ValidationError) as exc_info:
            TypeAdapter(IdentityFacetBinding).validate_python(
                {"state": "value", "ref": "x", "request": {"currency": "USD", "prices": "current"}}
            )
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "union_tag_not_found"


class TestIdentityUnitRequestBaseYearStrictness:
    """`IdentityUnitRequest.base_year` is `StrictInt`, matching
    `dfo_spec_kit.identity.core._require_integral_number`'s own exclusion
    of `bool` and numeric strings -- pure model-level coverage, then the
    wire-level door proving the resulting ValidationError reaches the same
    problem-details shape every other malformed identity gets."""

    def test_bool_base_year_is_rejected(self):
        from pydantic import ValidationError

        from search.models import IdentityUnitRequest

        with pytest.raises(ValidationError):
            IdentityUnitRequest(currency="USD", prices="constant", base_year=True)

    def test_numeric_string_base_year_is_rejected(self):
        from pydantic import ValidationError

        from search.models import IdentityUnitRequest

        with pytest.raises(ValidationError):
            IdentityUnitRequest(currency="USD", prices="constant", base_year="2023")

    def test_int_base_year_is_accepted_unchanged(self):
        from search.models import IdentityUnitRequest

        request = IdentityUnitRequest(currency="USD", prices="constant", base_year=2023)
        assert request.base_year == 2023

    def test_bool_base_year_on_resubmit_facets_is_request_invalid_400(self):
        """The same door as test_unit_request_lowercase_currency_is_request_invalid_400
        (TestResubmitIdentityRefusals, above), carrying a `base_year: true` unit
        request instead -- confirms the StrictInt refusal reaches the wire as
        the ordinary malformed-input 400, with no new error mapping needed."""
        resp = make_client().post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "dataset_resubmit",
                    "entries": [
                        {
                            "entry_kind": "identity",
                            "variable_text": "a resubmitted identity",
                            "identity": {
                                "shape": {
                                    "kind": "statvar_schema",
                                    "population_type": "some_shape",
                                },
                                "slots": [],
                                "entities": {},
                                "requested_window": {"kind": "all"},
                                "facets": {
                                    "unit": {
                                        "state": "value",
                                        "request": {
                                            "currency": "USD",
                                            "prices": "constant",
                                            "base_year": True,
                                        },
                                    }
                                },
                            },
                        }
                    ],
                }
            },
        )
        _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)


class TestInfraRefusals:
    def test_upstream_4xx_is_upstream_error_502(self):
        class _UpstreamErrorGraph(FakeGraph):
            def __init__(self):
                super().__init__()

            def detect_svs(self, query):
                from search.engine.errors import GraphInfraError

                raise GraphInfraError("not found", upstream_status=404)

        graph = _UpstreamErrorGraph()
        app = search_app(
            graph=graph,
            resolver=make_resolver(graph),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
        )
        client = TestClient(app, raise_server_exceptions=False)
        with patch("search.engine.extract.date") as mock_date:
            mock_date.today.return_value = PINNED_DATE
            mock_date.side_effect = None
            resp = client.post(
                "/api/search/resolve",
                json={
                    "input": {"kind": "raw_text", "query": "health ODA grants from USA to Ethiopia"}
                },
            )
        _assert_problem_shape(resp, status=502, code="upstream_error", retryable=True)

    def test_transient_infra_failure_is_service_unavailable_503(self):
        class _ErrorGraph(FakeGraph):
            def __init__(self):
                super().__init__()

            def detect_svs(self, query):
                from search.engine.errors import GraphInfraError

                raise GraphInfraError("boom")

        graph = _ErrorGraph()
        app = search_app(
            graph=graph,
            resolver=make_resolver(graph),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
        )
        client = TestClient(app, raise_server_exceptions=False)
        with patch("search.engine.extract.date") as mock_date:
            mock_date.today.return_value = PINNED_DATE
            mock_date.side_effect = None
            resp = client.post(
                "/api/search/resolve",
                json={
                    "input": {"kind": "raw_text", "query": "health ODA grants from USA to Ethiopia"}
                },
            )
        _assert_problem_shape(resp, status=503, code="service_unavailable", retryable=True)

    def test_deadline_exceeded_is_timeout_504(self, monkeypatch):

        async def _raise_timeout(*args, **kwargs):
            raise TimeoutError()

        monkeypatch.setattr("search.engine.app.resolve_async", _raise_timeout)
        resp = make_client().post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "health ODA"}},
        )
        _assert_problem_shape(resp, status=504, code="timeout", retryable=True)


class TestProblemDetailsTruncated:
    """ProblemDetails.truncated: present iff alternatives is present -- the
    contract a future caller populating alternatives will rely on (no
    caller does yet)."""

    def test_alternatives_and_truncated_round_trip(self):
        from dfo_platform import Alternative, ProblemDetails

        problem = ProblemDetails(
            type="/codes/request_invalid",
            title="Bad Request",
            status=400,
            detail="Bad request",
            code="request_invalid",
            alternatives=[Alternative(ref="a", label="A"), Alternative(ref="b", label="B")],
            truncated=True,
            trace_id="a" * 32,
            capability_version=settings().capability_version,
            retryable=False,
        )
        body = problem.to_json()
        assert body["alternatives"] == [{"ref": "a", "label": "A"}, {"ref": "b", "label": "B"}]
        assert body["truncated"] is True

    def test_truncated_omitted_when_no_alternatives(self):
        from dfo_platform import ProblemDetails

        problem = ProblemDetails(
            type="/codes/request_invalid",
            title="Bad Request",
            status=400,
            detail="Bad request",
            code="request_invalid",
            trace_id="a" * 32,
            capability_version=settings().capability_version,
            retryable=False,
        )
        body = problem.to_json()
        assert "alternatives" not in body
        assert "truncated" not in body


class TestUnhandledException:
    """A genuinely unclassified exception (neither EngineInfraError,
    EngineInputError, nor asyncio.TimeoutError) still returns the full RFC 9457
    problem-details shape on both transports -- code="internal_error", 500,
    retryable=false."""

    def test_unhandled_exception_returns_full_problem_details_500(self, monkeypatch):
        async def _raise_generic(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr("search.engine.app.resolve_async", _raise_generic)
        resp = make_client().post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "health ODA"}},
        )
        _assert_problem_shape(resp, status=500, code="internal_error", retryable=False)

    def test_sse_unclassified_exception_matches_internal_error_shape(self, monkeypatch):
        async def _raising_stream(request, *, graph=None, llm=None, **_kwargs):
            raise RuntimeError("boom")
            yield  # pragma: no cover — makes this a generator function

        monkeypatch.setattr("search.engine.app.resolve_events", _raising_stream)
        resp = make_client().post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "test"}},
            headers={"Accept": "text/event-stream"},
        )
        assert resp.status_code == 200
        frames = _parse_sse(resp.text)
        assert [ev for ev, _ in frames] == ["error"]
        body = frames[0][1]
        assert body["type"] == "/codes/internal_error"
        assert body["status"] == 500
        assert body["code"] == "internal_error"
        assert body["retryable"] is False
        assert isinstance(body["trace_id"], str) and body["trace_id"]


class TestOptionsDoorProblemShape:
    """The options door (POST /api/search/options) emits the same shared
    problem-details body shape as every other door -- no bespoke handler."""

    def test_malformed_body_is_problem_json(self):
        resp = make_client().post("/api/search/options", json={})
        _assert_problem_shape(resp, status=400, code="request_invalid", retryable=False)


class TestTraceIdSharing:
    def test_trace_id_present_in_success_diagnostics(self):
        resp = make_client().post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "   "}},
        )
        assert resp.status_code == 200
        body = resp.json()
        trace_id = body["diagnostics"]["trace_id"]
        assert isinstance(trace_id, str) and len(trace_id) == 32

    def test_trace_id_differs_across_requests(self):
        client = make_client()
        r1 = client.post(
            "/api/search/resolve", json={"input": {"kind": "raw_text", "query": "   "}}
        )
        r2 = client.post(
            "/api/search/resolve", json={"input": {"kind": "raw_text", "query": "   "}}
        )
        assert r1.json()["diagnostics"]["trace_id"] != r2.json()["diagnostics"]["trace_id"]

    def test_trace_id_shared_between_response_and_done_sse_events(self):
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "   "}},
            headers={"Accept": "text/event-stream"},
        )
        assert resp.status_code == 200
        frames = _parse_sse(resp.text)
        response_frame = next(data for ev, data in frames if ev == "response")
        done_frame = next(data for ev, data in frames if ev == "done")
        response_trace_id = response_frame["response"]["diagnostics"]["trace_id"]
        assert response_trace_id == done_frame["trace_id"]


class TestTraceIdConcurrency:
    """Two genuinely interleaved requests get distinct trace_ids:
    engine.trace's ContextVar is per-asyncio-Task, and Starlette gives each
    request its own task, so this pins that isolation under real
    concurrency, not just two sequential TestClient calls (which never
    actually overlap).
    """

    async def _post(self, client: httpx.AsyncClient, query: str) -> httpx.Response:
        return await client.post(
            "/api/search/resolve", json={"input": {"kind": "raw_text", "query": query}}
        )

    def test_concurrent_buffered_requests_get_distinct_trace_ids(self):
        async def scenario() -> tuple[httpx.Response, httpx.Response]:
            app = search_app(
                graph=FakeGraph(),
                resolver=make_resolver(FakeGraph()),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
            )
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await asyncio.gather(
                    self._post(client, "   "),
                    self._post(client, "   "),
                )

        r1, r2 = asyncio.run(scenario())
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["diagnostics"]["trace_id"] != r2.json()["diagnostics"]["trace_id"]

    def test_concurrent_sse_streams_get_distinct_and_self_consistent_trace_ids(self):
        async def _one_stream(client: httpx.AsyncClient, query: str) -> str:
            async with client.stream(
                "POST",
                "/api/search/resolve",
                json={"input": {"kind": "raw_text", "query": query}},
                headers={"Accept": "text/event-stream"},
            ) as resp:
                return (await resp.aread()).decode()

        async def scenario() -> tuple[str, str]:
            app = search_app(
                graph=FakeGraph(),
                resolver=make_resolver(FakeGraph()),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
            )
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await asyncio.gather(
                    _one_stream(client, "   "),
                    _one_stream(client, "  "),
                )

        body_1, body_2 = asyncio.run(scenario())
        frames_1 = _parse_sse(body_1)
        frames_2 = _parse_sse(body_2)

        def _trace_ids(frames: list[tuple[str, dict]]) -> tuple[str, str]:
            response_frame = next(data for ev, data in frames if ev == "response")
            done_frame = next(data for ev, data in frames if ev == "done")
            return response_frame["response"]["diagnostics"]["trace_id"], done_frame["trace_id"]

        response_trace_1, done_trace_1 = _trace_ids(frames_1)
        response_trace_2, done_trace_2 = _trace_ids(frames_2)

        # Each stream's own response/done pair agrees...
        assert response_trace_1 == done_trace_1
        assert response_trace_2 == done_trace_2
        # ...and the two streams never share a trace_id.
        assert response_trace_1 != response_trace_2


class TestSSEErrorEvent:
    """The SSE terminal `error` event is the same RFC 9457 problem-details
    shape a non-2xx failure returns (ErrorEvent subclasses ProblemDetails
    verbatim -- see events.py)."""

    def test_error_event_carries_full_problem_details_shape(self, monkeypatch):
        async def _raising_stream(request, *, graph=None, llm=None, **_kwargs):
            from search.engine.errors import GraphInfraError

            raise GraphInfraError("not found", upstream_status=404)
            yield  # pragma: no cover — makes this a generator function

        monkeypatch.setattr("search.engine.app.resolve_events", _raising_stream)
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "test"}},
            headers={"Accept": "text/event-stream"},
        )
        assert resp.status_code == 200
        frames = _parse_sse(resp.text)
        assert [ev for ev, _ in frames] == ["error"]
        body = frames[0][1]
        assert body["type"] == "/codes/upstream_error"
        assert body["title"]
        assert body["status"] == 502
        assert body["detail"] == "Bad gateway: upstream returned an error"
        assert body["code"] == "upstream_error"
        assert body["retryable"] is True
        assert isinstance(body["trace_id"], str) and body["trace_id"]
        assert body["capability_version"] == settings().capability_version
        assert "pointer" not in body
        assert "alternatives" not in body
        assert "truncated" not in body
