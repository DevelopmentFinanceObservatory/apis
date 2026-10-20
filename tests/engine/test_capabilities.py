"""Tests that published capabilities, version stamps, and door liveness match runtime behaviour.
Code routes must resolve every registered code.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import get_args

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from code_index.shared import get_index
from dfo_platform.build import build_commit
from search.engine import config, schema
from search.engine.capabilities import build_capabilities
from search.engine.codes import CODE_CATALOG
from search.engine.config import (
    SEARCH_MAX_ENTITY_TEXT_CHARS,
    SEARCH_MAX_PART_ENTITIES,
    SEARCH_MAX_PARTS,
    SEARCH_MAX_ROLE_TEXT_CHARS,
    SEARCH_MAX_SOURCE_TEXT_CHARS,
    SEARCH_MAX_TIME_TEXT_CHARS,
    SEARCH_MAX_UNIT_TEXT_CHARS,
    SEARCH_MAX_VARIABLE_TEXT_CHARS,
    settings,
)
from search.engine.schema import schema_payload
from search.models import SCHEMA_VERSION, Axis, FacetAxis
from search.observations import observation_settings
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, make_resolver


def make_client() -> TestClient:
    # Bare FakeSdmxClient() loads a real Health-ODAGrants/USA/Ethiopia facet
    # (df-01's combo) -- TestCapabilityVersionAgreesEverywhere posts that exact
    # dev-finance query and needs a "definite" resolve, not just a live-network
    # non-crash.
    app = search_app(
        graph=FakeGraph(),
        resolver=make_resolver(FakeGraph()),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(),
    )
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def capabilities_body() -> dict:
    resp = make_client().get("/api/search/capabilities")
    assert resp.status_code == 200
    return resp.json()


class TestCapabilitiesShape:
    def test_capability_version_distinct_from_schema_version(self, capabilities_body):
        # Distinct from schema_version's bare "1"/"1.0" literal.
        assert capabilities_body["capability_version"] != SCHEMA_VERSION
        assert capabilities_body["capability_version"] == settings().capability_version

    def test_build(self, capabilities_body):
        assert capabilities_body["build"] == build_commit()

    def test_axes_match_the_contract_frozen_enum(self, capabilities_body):
        assert sorted(capabilities_body["axes"]) == sorted(get_args(Axis))

    def test_facet_axes_match_facet_axis_vocabulary_and_are_sorted(self, capabilities_body):
        assert capabilities_body["facet_axes"] == sorted(get_args(FacetAxis))

    def test_recipe_classes_lists_the_conversion_recipe_members(self, capabilities_body):
        # Sourced from materialise.recipe_bodies.RECIPE_CLASSES, the sorted
        # keys of the deployed recipe bodies.
        assert capabilities_body["recipe_classes"] == [
            "ConversionRecipeDAC",
            "ConversionRecipeIDS",
            "ConversionRecipeIMF",
        ]

    def test_doors_cover_every_resolve_door_plus_options(self, capabilities_body):
        doors = {d["name"]: d["live"] for d in capabilities_body["doors"]}
        assert doors == {
            "raw_text": True,
            "parsed": True,
            "dataset_resubmit": True,
            "options": True,
        }

    def test_default_policies_reflect_pac_default(self, capabilities_body):
        policies = {p["name"]: p for p in capabilities_body["default_policies"]}
        assert policies["place_as_constraint"]["live"] is True
        assert policies["place_as_constraint"]["default"] == settings().seam_default
        # Unit completion declares every member: the configured currency,
        # current prices, and the conversion tables' base year.
        for name in ("unit_currency", "unit_prices", "unit_base_year"):
            assert policies[name]["live"] is True


class TestCodeIndexStamp:
    """The published `code_index` field must match the process-wide
    `CodeIndex` singleton, not a hand-copied value."""

    def test_code_index_matches_the_loaded_index(self, capabilities_body):
        stamp = get_index().stamp
        code_index = capabilities_body["code_index"]
        assert code_index["content_hash"] == stamp.content_hash
        assert code_index["exported_at"] == stamp.exported_at
        assert code_index["code_counts"] == dict(stamp.code_counts)


class TestPublishedCapsEqualEnforcedValues:
    """Published caps must equal the SAME constants the engine enforces --
    not a hand-copied list. Compares the wire body directly against the
    config constants the engine actually reads.
    """

    def test_caps_equal_config_constants(self, capabilities_body):
        caps = capabilities_body["caps"]
        assert caps["max_query_chars"] == settings().max_query_chars
        assert caps["max_candidates"] == settings().max_candidates
        assert caps["max_variables"] == settings().max_variables
        assert caps["obs_preview_points"] == observation_settings().obs_preview_points
        assert caps["obs_inline_max"] == observation_settings().obs_inline_max
        assert caps["role_bulk_rank_max_rows"] == observation_settings().role_bulk_rank_max_rows
        assert caps["max_options"] == settings().max_options
        assert caps["max_parts"] == SEARCH_MAX_PARTS
        assert caps["max_part_entities"] == SEARCH_MAX_PART_ENTITIES
        assert caps["max_variable_text_chars"] == SEARCH_MAX_VARIABLE_TEXT_CHARS
        assert caps["max_entity_text_chars"] == SEARCH_MAX_ENTITY_TEXT_CHARS
        assert caps["max_role_text_chars"] == SEARCH_MAX_ROLE_TEXT_CHARS
        assert caps["max_time_text_chars"] == SEARCH_MAX_TIME_TEXT_CHARS
        assert caps["max_unit_text_chars"] == SEARCH_MAX_UNIT_TEXT_CHARS
        assert caps["max_source_text_chars"] == SEARCH_MAX_SOURCE_TEXT_CHARS
        assert caps["region_max_members"] == settings().region_max_members
        assert caps["scope_min_year"] == settings().scope_min_year
        assert caps["scope_max_window_years"] == settings().scope_max_window_years

    def test_max_parts_is_distinct_from_max_variables(self, capabilities_body):
        # raw_text clamps an over-limit extraction (variables_clamped) and keeps
        # resolving; parsed refuses request_invalid on an over-limit parts list.
        # The two caps happen to start at the same number but must be able to
        # move independently, so they are published as two separate fields.
        caps = capabilities_body["caps"]
        assert caps["max_parts"] == SEARCH_MAX_PARTS
        assert caps["max_variables"] == settings().max_variables


def _renamed_ref_key(document: dict) -> None:
    graph_ref = document["response"]["$defs"]["GraphRef"]
    graph_ref["properties"] = {
        ("dcid" if key == "ref" else key): value for key, value in graph_ref["properties"].items()
    }
    graph_ref["required"] = ["dcid" if key == "ref" else key for key in graph_ref["required"]]


def _retyped_label(document: dict) -> None:
    document["response"]["$defs"]["GraphRef"]["properties"]["label"]["type"] = "integer"


def _reworded_description(document: dict) -> None:
    document["response"]["$defs"]["GraphRef"]["properties"]["label"]["description"] = "reworded"


_UNREADABLE_INDEX_PROBE = """
import code_index
import code_index.loader

def unreadable(source):
    raise FileNotFoundError("the code index was read")

code_index.load_index = code_index.loader.load_index = unreadable
import search.engine.config as config
print("imported")
config.settings().capability_version
print("computed")
"""


def test_the_capability_token_reads_no_code_index():
    """The token hashes the served schema, whose datasets come from the declarations, so
    importing the config and computing the token read no code index. The packaged index is
    unreadable in the probe, and no release is bound for `get_index()` to read."""
    result = subprocess.run(
        [sys.executable, "-c", _UNREADABLE_INDEX_PROBE],
        capture_output=True,
        text=True,
        env={**os.environ, "SEARCH_GRAPH_BASE": "https://graph.invalid/core/api"},
        check=False,
    )

    assert result.stdout.split() == ["imported", "computed"], result.stderr


class TestCapabilityVersionPerturbation:
    """A cap change must actually rotate capability_version -- not just be
    declared in CapabilityCaps. Proven by perturbing one of the parsed-door
    cap constants and recomputing the version function directly, rather than
    by auditing what the hash claims to cover.
    """

    @pytest.mark.parametrize(
        ("edit", "moves"),
        [
            pytest.param(_renamed_ref_key, True, id="renamed-field"),
            pytest.param(_retyped_label, True, id="retyped-field"),
            pytest.param(_reworded_description, False, id="reworded-description"),
        ],
    )
    def test_a_served_schema_change_moves_the_capability_version(self, monkeypatch, edit, moves):
        original = config.settings()._compute_capability_version()
        served = schema.schema_payload

        def edited() -> dict:
            document = served()
            edit(document)
            return document

        monkeypatch.setattr(schema, "schema_payload", edited)
        assert (config.settings()._compute_capability_version() != original) is moves

    def test_changing_a_new_cap_moves_the_capability_version(self, monkeypatch):
        original = config.settings()._compute_capability_version()
        monkeypatch.setattr(config, "SEARCH_MAX_PARTS", config.SEARCH_MAX_PARTS + 1)
        perturbed = config.settings()._compute_capability_version()
        assert perturbed != original

    def test_changing_each_new_cap_moves_the_capability_version(self, monkeypatch):
        # Every one of the eight parsed-door caps must individually be wired
        # into the hash -- not just the one spot-checked above. Fixed
        # constants, patched directly on the config module (SearchSettings'
        # own _capability_version_payload reads them as module globals, not
        # instance fields).
        new_cap_names = [
            "SEARCH_MAX_PARTS",
            "SEARCH_MAX_PART_ENTITIES",
            "SEARCH_MAX_VARIABLE_TEXT_CHARS",
            "SEARCH_MAX_ENTITY_TEXT_CHARS",
            "SEARCH_MAX_ROLE_TEXT_CHARS",
            "SEARCH_MAX_TIME_TEXT_CHARS",
            "SEARCH_MAX_UNIT_TEXT_CHARS",
            "SEARCH_MAX_SOURCE_TEXT_CHARS",
        ]
        baseline = config.settings()._compute_capability_version()
        for name in new_cap_names:
            with monkeypatch.context() as m:
                m.setattr(config, name, getattr(config, name) + 1)
                perturbed = config.settings()._compute_capability_version()
            assert perturbed != baseline, f"{name} is not wired into the capability_version hash"

    def test_changing_each_role_scope_cap_moves_the_capability_version(self):
        # The role-scope discovery caps must each be individually wired
        # into the hash too, the same as the parsed-door caps above. These
        # are SearchSettings fields, so each is perturbed on a model_copy
        # rather than patched on the config module.
        role_scope_field_names = [
            "region_max_members",
            "scope_min_year",
            "scope_max_window_years",
        ]
        base = config.settings()
        baseline = base._compute_capability_version()
        for field in role_scope_field_names:
            perturbed_settings = base.model_copy(update={field: getattr(base, field) + 1})
            perturbed = perturbed_settings._compute_capability_version()
            assert perturbed != baseline, f"{field} is not wired into the capability_version hash"

    @pytest.mark.parametrize("cap", ["obs_inline_max", "role_bulk_rank_max_rows"])
    def test_changing_an_observation_cap_moves_the_capability_version(self, search_settings, cap):
        original = config.settings().capability_version
        search_settings(**{cap: getattr(observation_settings(), cap) + 1})
        assert config.settings().capability_version != original

    def test_changing_the_unit_ref_source_marker_moves_the_capability_version(self, monkeypatch):
        # The unit-ref answer is a graph read
        # (search.engine.unit.refs, EngineGraphClient.currency_units()) that
        # cannot run at import time, so it cannot be hashed directly -- see
        # config.py's own comment on SEARCH_UNIT_REF_SOURCE. These two fixed
        # markers stand in for it and must each individually be wired into
        # the hash, the same as the parsed-door/role-scope caps above.
        original = config.settings()._compute_capability_version()
        monkeypatch.setattr(config, "SEARCH_UNIT_REF_SOURCE", "graph:currency_units_v2")
        perturbed = config.settings()._compute_capability_version()
        assert perturbed != original

        monkeypatch.undo()
        restored = config.settings()._compute_capability_version()
        assert restored == original

    def test_changing_the_unit_ref_prices_vocabulary_moves_the_capability_version(
        self, monkeypatch
    ):
        original = config.settings()._compute_capability_version()
        monkeypatch.setattr(config, "SEARCH_UNIT_REF_PRICES_VOCABULARY", ("current", "constant"))
        perturbed = config.settings()._compute_capability_version()
        assert perturbed != original

        monkeypatch.undo()
        restored = config.settings()._compute_capability_version()
        assert restored == original

    def test_changing_the_recipe_classes_moves_the_capability_version(self, monkeypatch):
        original = config.settings()._compute_capability_version()
        monkeypatch.setattr(config, "SEARCH_RECIPE_CLASSES", ("ConversionRecipeDAC",))
        perturbed = config.settings()._compute_capability_version()
        assert perturbed != original

        monkeypatch.undo()
        restored = config.settings()._compute_capability_version()
        assert restored == original


class TestPerDoorLivenessMatchesEnforcement:
    """The published liveness flag for each door must match what /resolve
    actually does for that door -- not just what the document claims.
    """

    def test_parsed_door_marked_live_and_actually_resolves(self, capabilities_body):
        doors = {d["name"]: d["live"] for d in capabilities_body["doors"]}
        assert doors["parsed"] is True

        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={"input": {"kind": "parsed", "parts": [{"variable_text": "x"}]}},
        )
        assert resp.status_code == 200

    def test_raw_text_door_marked_live_and_actually_resolves(self, capabilities_body):
        doors = {d["name"]: d["live"] for d in capabilities_body["doors"]}
        assert doors["raw_text"] is True

        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "health ODA grants from USA to Ethiopia"}},
        )
        assert resp.status_code == 200

    def test_options_door_advertised_and_reachable(self, capabilities_body):
        # Callers route on liveness, so this door must be advertised and mounted.
        doors = {d["name"]: d["live"] for d in capabilities_body["doors"]}
        assert doors["options"] is True

        client = make_client()
        assert client.post("/api/search/options", json={}).status_code != 404


class TestCapabilityVersionAgreesEverywhere:
    """/capabilities, a problem body, and a resolved.trace must all carry the
    identical capability_version -- the same token, not merely the same shape.
    """

    def test_agrees_with_a_problem_body(self, capabilities_body):
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "x" * (settings().max_query_chars + 1)}},
        )
        assert resp.status_code == 422
        assert resp.json()["capability_version"] == capabilities_body["capability_version"]

    def test_agrees_with_a_resolved_trace(self, capabilities_body):
        client = make_client()
        resp = client.post(
            "/api/search/resolve",
            json={"input": {"kind": "raw_text", "query": "fertility rate in Kenya"}},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "definite"
        trace_version = body["parts"][0]["spec"]["resolved"]["trace"]["capability_version"]
        assert trace_version == capabilities_body["capability_version"]


class TestCodesEndpoint:
    def test_known_code_dereferences(self):
        client = make_client()
        resp = client.get("/codes/query_too_long")
        assert resp.status_code == 200
        body = resp.json()
        title, detail = CODE_CATALOG["query_too_long"]
        assert body["code"] == "query_too_long"
        assert body["title"] == title
        assert body["detail"] == detail

    def test_every_problem_type_uri_dereferences(self):
        # Every code in CODE_CATALOG is a real problem `type` target
        # (ProblemDetails.type == f"/codes/{code}"); each must resolve.
        client = make_client()
        for code in CODE_CATALOG:
            resp = client.get(f"/codes/{code}")
            assert resp.status_code == 200, code

    def test_flagged_match_gate_codes_dereference(self):
        """NoData.gate values are drawn from this same served registry (see
        codes.py's SEARCH_REFUSAL_CODES docstring). Each must dereference
        here like any other code."""
        client = make_client()
        for code in (
            "only_alternate_has_data",
            "no_member_has_data_at_place",
            "no_member_asserts_bound_properties",
        ):
            resp = client.get(f"/codes/{code}")
            assert resp.status_code == 200, code
            title, detail = CODE_CATALOG[code]
            body = resp.json()
            assert body["code"] == code
            assert body["title"] == title
            assert body["detail"] == detail

    def test_unknown_code_is_404_problem_shape(self):
        client = make_client()
        resp = client.get("/codes/not_a_real_code")
        assert resp.status_code == 404
        assert resp.headers["content-type"].startswith("application/problem+json")
        body = resp.json()
        assert body["code"] == "code_not_found"
        assert body["type"] == "/codes/code_not_found"


class TestSchemaAndCapabilitiesAreFrozenDocuments:
    """`/api/search/schema` and `/api/search/capabilities` are each served by
    a `dfo_platform.static_response.FrozenJSON` built once per surface (see
    `search.engine.app.search_surface`'s `freeze_public_json` call sites), so both
    routes carry an ETag and `Cache-Control`, honor gzip negotiation and
    `If-None-Match`, and serve exactly the same JSON a plain `JSONResponse`
    of the underlying payload would.
    """

    @pytest.mark.parametrize("path", ["/api/search/schema", "/api/search/capabilities"])
    def test_headers_present_on_plain_get(self, path):
        client = make_client()
        resp = client.get(path, headers={"accept-encoding": "identity"})
        assert resp.status_code == 200
        assert resp.headers["cache-control"] == "public, max-age=300"
        assert resp.headers["vary"] == "Accept-Encoding, Origin"
        assert "etag" in resp.headers
        assert "content-encoding" not in resp.headers

    @pytest.mark.parametrize("path", ["/api/search/schema", "/api/search/capabilities"])
    def test_gzip_round_trips_to_the_same_json(self, path):
        # httpx transparently decodes a gzip Content-Encoding, so `.content`
        # below is already the inflated body -- this exercises the server's
        # gzip representation without hand-decompressing it.
        client = make_client()
        identity = client.get(path, headers={"accept-encoding": "identity"})
        gzipped = client.get(path, headers={"accept-encoding": "gzip"})
        assert gzipped.headers["content-encoding"] == "gzip"
        assert gzipped.content == identity.content
        assert gzipped.json() == identity.json()

    @pytest.mark.parametrize("path", ["/api/search/schema", "/api/search/capabilities"])
    def test_304_on_if_none_match(self, path):
        client = make_client()
        first = client.get(path, headers={"accept-encoding": "identity"})
        second = client.get(
            path,
            headers={"accept-encoding": "identity", "if-none-match": first.headers["etag"]},
        )
        assert second.status_code == 304
        assert second.content == b""

    def test_schema_identity_body_matches_jsonresponse_of_schema_payload(self):
        client = make_client()
        resp = client.get("/api/search/schema", headers={"accept-encoding": "identity"})
        assert resp.content == JSONResponse(schema_payload()).body

    def test_capabilities_identity_body_matches_jsonresponse_of_build_capabilities(self):
        client = make_client()
        resp = client.get("/api/search/capabilities", headers={"accept-encoding": "identity"})
        assert resp.content == JSONResponse(build_capabilities().model_dump(mode="json")).body
