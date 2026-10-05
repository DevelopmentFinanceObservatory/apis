"""Perturbation tests verify which inputs affect the graph capability token."""

from __future__ import annotations

import os
import subprocess
import sys

from pydantic import TypeAdapter

from dfo_platform import HealthResponse
from graph_api import version
from graph_api.capabilities import build_capabilities
from graph_api.codes import GRAPH_CODE_CATALOGUE
from graph_api.node_models import NodeResponse
from graph_api.resolve_models import ResolveRequest, ResolveResponse
from graph_api.source_models import SourceCatalogue, SourceDocumentation


def _baseline_capability_document() -> dict:
    return build_capabilities(
        capability_version=version._EXEMPLAR_CAPABILITY_VERSION,
        build=version._EXEMPLAR_BUILD,
    ).model_dump(mode="json")


def _baseline_wire_schemas() -> dict[str, dict]:
    """Return the default wire-schema set hashed by the capability version."""
    return {
        "health_response": HealthResponse.model_json_schema(),
        "resolve_request": ResolveRequest.model_json_schema(),
        "resolve_response": ResolveResponse.model_json_schema(),
        "node_response": TypeAdapter(NodeResponse).json_schema(),
        "source_documentation_response": SourceDocumentation.model_json_schema(),
        "source_catalogue_response": SourceCatalogue.model_json_schema(),
    }


class TestPerturbationsMoveTheToken:
    def test_node_pool_size_change_moves_the_token(self):
        baseline = version._compute_graph_capability_version()
        perturbed_document = _baseline_capability_document()
        perturbed_document["caps"]["node_pool_size"] += 1

        perturbed = version._compute_graph_capability_version(
            capability_document=perturbed_document
        )

        assert perturbed != baseline

    def test_resolve_timeout_seconds_change_moves_the_token(self):
        baseline = version._compute_graph_capability_version()
        perturbed_document = _baseline_capability_document()
        perturbed_document["caps"]["resolve_timeout_seconds"] += 1

        perturbed = version._compute_graph_capability_version(
            capability_document=perturbed_document
        )

        assert perturbed != baseline

    def test_node_timeout_seconds_change_moves_the_token(self):
        baseline = version._compute_graph_capability_version()
        perturbed_document = _baseline_capability_document()
        perturbed_document["caps"]["node_timeout_seconds"] += 1

        perturbed = version._compute_graph_capability_version(
            capability_document=perturbed_document
        )

        assert perturbed != baseline

    def test_max_dcid_chars_change_moves_the_token(self):
        baseline = version._compute_graph_capability_version()
        perturbed_document = _baseline_capability_document()
        perturbed_document["caps"]["max_dcid_chars"] += 1

        perturbed = version._compute_graph_capability_version(
            capability_document=perturbed_document
        )

        assert perturbed != baseline

    def test_door_liveness_flip_moves_the_token(self):
        baseline = version._compute_graph_capability_version()
        perturbed_document = _baseline_capability_document()
        perturbed_document["doors"][0]["live"] = False

        perturbed = version._compute_graph_capability_version(
            capability_document=perturbed_document
        )

        assert perturbed != baseline

    def test_code_catalogue_title_change_moves_the_token(self):
        baseline = version._compute_graph_capability_version()
        code, (title, detail) = next(iter(sorted(GRAPH_CODE_CATALOGUE.items())))
        mutated_catalogue = dict(GRAPH_CODE_CATALOGUE)
        mutated_catalogue[code] = (title + " (mutated)", detail)

        perturbed = version._compute_graph_capability_version(code_catalogue=mutated_catalogue)

        assert perturbed != baseline

    def test_wire_schema_property_change_moves_the_token(self):
        baseline = version._compute_graph_capability_version()
        mutated_schemas = _baseline_wire_schemas()
        mutated_schemas["health_response"] = {
            **mutated_schemas["health_response"],
            "title": mutated_schemas["health_response"].get("title", ""),
            "required": [*mutated_schemas["health_response"].get("required", []), "mutated"],
        }

        perturbed = version._compute_graph_capability_version(wire_schemas=mutated_schemas)

        assert perturbed != baseline

    def test_source_catalogue_response_schema_change_moves_the_token(self):
        """`SourceCatalogue` is the sixth hashed wire schema; changing only
        its schema must move the token.
        """
        baseline = version._compute_graph_capability_version()
        mutated_schemas = _baseline_wire_schemas()
        mutated_schemas["source_catalogue_response"] = {
            **mutated_schemas["source_catalogue_response"],
            "title": mutated_schemas["source_catalogue_response"].get("title", ""),
            "required": [
                *mutated_schemas["source_catalogue_response"].get("required", []),
                "mutated",
            ],
        }

        perturbed = version._compute_graph_capability_version(wire_schemas=mutated_schemas)

        assert perturbed != baseline

    def test_wire_schema_addition_moves_the_token(self):
        baseline = version._compute_graph_capability_version()
        mutated_schemas = _baseline_wire_schemas()
        mutated_schemas["extra_schema"] = {"type": "object"}

        perturbed = version._compute_graph_capability_version(wire_schemas=mutated_schemas)

        assert perturbed != baseline


class TestNonPerturbationsDoNotMoveTheToken:
    def test_wire_schema_description_edit_does_not_move_the_token(self):
        baseline = version._compute_graph_capability_version()
        mutated_schemas = _baseline_wire_schemas()
        # Mutate a description -- stripped before hashing
        # (_strip_wire_invisible_keys), so this must be a no-op on the token.
        mutated_schemas["health_response"] = {
            **mutated_schemas["health_response"],
            "description": "a reworded, purely cosmetic description",
        }

        unchanged = version._compute_graph_capability_version(wire_schemas=mutated_schemas)

        assert unchanged == baseline

    def test_build_id_change_does_not_move_the_token(self):
        document_a = build_capabilities(
            capability_version="graph_whatever", build="build-A"
        ).model_dump(mode="json")
        document_b = build_capabilities(
            capability_version="graph_whatever", build="build-B"
        ).model_dump(mode="json")

        token_a = version._compute_graph_capability_version(capability_document=document_a)
        token_b = version._compute_graph_capability_version(capability_document=document_b)

        assert token_a == token_b


def test_baseline_wire_schemas_mirror_matches_the_real_default():
    """The helper's keys and schemas must match the version function's defaults."""
    baseline = version._compute_graph_capability_version()
    mirrored = version._compute_graph_capability_version(wire_schemas=_baseline_wire_schemas())

    assert mirrored == baseline


def test_graph_capability_version_is_graph_prefixed_and_deterministic():
    assert version.GRAPH_CAPABILITY_VERSION.startswith("graph_")
    assert version._compute_graph_capability_version() == version.GRAPH_CAPABILITY_VERSION


def test_graph_capability_version_is_stable_across_process_runs():
    """The token is deterministic across subprocesses with hash seeds 1 and 2."""
    code = "from graph_api.version import GRAPH_CAPABILITY_VERSION as v\nprint(v)\n"
    first = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "1"},
    )
    second = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "2"},
    )
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first.stdout.strip() == second.stdout.strip()
