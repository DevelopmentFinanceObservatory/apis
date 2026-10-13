"""The capabilities response publishes the constants enforced by this surface."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from code_index.shared import get_index
from dfo_platform.build import build_commit
from graph_api import version
from graph_api.capabilities import Capabilities, DataBuilds, build_capabilities
from graph_api.config import (
    GRAPH_NODE_POOL_SIZE,
    GRAPH_NODE_TIMEOUT_SECONDS,
    GRAPH_RESOLVE_TIMEOUT_SECONDS,
)
from graph_api.deps import GRAPH_DEPS_SLOT, GraphDeps
from graph_api.node_models import NODES_MAX_DCIDS
from graph_api.routing import DCID_MAX_LENGTH
from graph_api.version import GRAPH_CAPABILITY_VERSION
from graph_generation.fake import FakeGeneration
from tests.origin_catalogue.fake_catalogue import restrictive_catalogue

from .conftest import FakeGraphNodeReader, make_app, make_client


def _capabilities() -> Capabilities:
    return build_capabilities(capability_version=GRAPH_CAPABILITY_VERSION, build=build_commit())


class TestPublishedCapsEqualEnforcedValues:
    def test_caps_equal_config_constants(self):
        caps = _capabilities().caps
        assert caps.node_pool_size == GRAPH_NODE_POOL_SIZE
        assert caps.resolve_timeout_seconds == GRAPH_RESOLVE_TIMEOUT_SECONDS
        assert caps.node_timeout_seconds == GRAPH_NODE_TIMEOUT_SECONDS
        assert caps.max_dcid_chars == DCID_MAX_LENGTH

    def test_capability_version_and_build_are_the_real_computed_values(self):
        document = _capabilities()
        assert document.capability_version == GRAPH_CAPABILITY_VERSION
        assert document.build == build_commit()


class TestDoorLivenessAndOtherFields:
    def test_all_doors_are_live(self):
        doors = {d.name: d.live for d in _capabilities().doors}
        assert doors == {
            "resolve": True,
            "node": True,
            "members": True,
            "roots": True,
            "sources": True,
            "schema": True,
        }

    def test_node_shape_gaps_are_the_paths_every_carrying_node_gaps(self):
        assert _capabilities().node_shape_gaps == ["stat_var.definition.measure.definition"]

    def test_source_document_gaps_match_the_reader_s_own_unconditional_gaps(self):
        from materialise.provenance_doc import UNCONDITIONAL_GAPS

        assert _capabilities().source_document_gaps == list(UNCONDITIONAL_GAPS)

    def test_source_document_gaps_are_the_members_no_source_fills(self):
        assert _capabilities().source_document_gaps == [
            "scope_and_units.countries",
            "usage.api_access",
            "usage.python",
        ]

    def test_the_source_catalogue_is_the_curated_registry(self):
        assert _capabilities().source_catalogue_curated is True


class TestEndpoint:
    def test_returns_200_with_the_real_computed_document(self):
        client = make_client()

        resp = client.get("/api/graph/capabilities")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["capability_version"] == GRAPH_CAPABILITY_VERSION
        assert body["build"] == build_commit()
        assert body["caps"] == {
            "node_pool_size": GRAPH_NODE_POOL_SIZE,
            "resolve_timeout_seconds": GRAPH_RESOLVE_TIMEOUT_SECONDS,
            "node_timeout_seconds": GRAPH_NODE_TIMEOUT_SECONDS,
            "max_dcid_chars": DCID_MAX_LENGTH,
            "max_node_batch_dcids": NODES_MAX_DCIDS,
        }

    def test_serves_public_with_an_etag_and_answers_304(self):
        client = _with_deps(FakeGeneration(token="rev-1"), "coverage-1")
        first = client.get("/api/graph/capabilities")

        again = client.get(
            "/api/graph/capabilities", headers={"if-none-match": first.headers["etag"]}
        )

        assert first.headers["cache-control"] == "public, max-age=300"
        assert first.headers["etag"]
        assert again.status_code == 304


def _with_deps(
    generation: FakeGeneration, coverage_build: str | None, *, coverage_load_failed: bool = False
):
    """A surface whose published deps carry this generation and coverage build."""
    app = make_app()
    coverage = (
        None
        if coverage_build is None
        else SimpleNamespace(manifest=SimpleNamespace(build_id=coverage_build))
    )
    GRAPH_DEPS_SLOT.set(
        app,
        GraphDeps(
            graph=FakeGraphNodeReader(),
            resolver=None,
            dcp_executor=None,
            generation=generation,
            provenance_docs=None,
            coverage=coverage,
            coverage_load_failed=coverage_load_failed,
        ),
    )
    return TestClient(app)


class TestDataBuilds:
    def test_publishes_each_data_build_the_process_serves(self, serve_catalogue):
        serve_catalogue(restrictive_catalogue(build_id="catalogue-1"))
        client = _with_deps(FakeGeneration(token="rev-1"), "coverage-1")

        body = client.get("/api/graph/capabilities").json()

        assert body["data_builds"] == {
            "graph_generation": "rev-1",
            "origin_catalogue": "catalogue-1",
            "coverage_index": "coverage-1",
            "code_index": get_index().stamp.content_hash,
        }

    def test_unpublished_deps_serve_null_generation_and_coverage(self, serve_catalogue):
        serve_catalogue(restrictive_catalogue(build_id="catalogue-1"))

        body = make_client().get("/api/graph/capabilities").json()

        assert body["data_builds"]["graph_generation"] is None
        assert body["data_builds"]["coverage_index"] is None
        assert body["data_builds"]["origin_catalogue"] == "catalogue-1"

    def test_a_transient_gap_in_a_build_is_not_cached(self):
        client = _with_deps(FakeGeneration(token="rev-1"), None, coverage_load_failed=True)
        unknown_generation = _with_deps(FakeGeneration(), "coverage-1")

        responses = [
            make_client().get("/api/graph/capabilities"),
            client.get("/api/graph/capabilities"),
            unknown_generation.get("/api/graph/capabilities"),
        ]

        assert [r.headers["cache-control"] for r in responses] == ["no-store"] * 3
        assert all("etag" not in r.headers for r in responses)

    def test_unconfigured_coverage_is_a_permanent_null_and_is_cached(self):
        resp = _with_deps(FakeGeneration(token="rev-1"), None).get("/api/graph/capabilities")

        assert resp.json()["data_builds"]["coverage_index"] is None
        assert resp.headers["cache-control"] == "public, max-age=300"
        assert resp.headers["etag"]

    def test_a_build_change_moves_the_etag_and_leaves_the_token(self):
        generation = FakeGeneration(token="rev-1")
        client = _with_deps(generation, "coverage-1")
        before = client.get("/api/graph/capabilities")

        generation.advance("rev-2")
        after = client.get("/api/graph/capabilities")

        assert before.json()["data_builds"]["graph_generation"] == "rev-1"
        assert after.json()["data_builds"]["graph_generation"] == "rev-2"
        assert before.headers["etag"] != after.headers["etag"]
        assert (
            before.json()["capability_version"]
            == after.json()["capability_version"]
            == version._compute_graph_capability_version()
        )

    def test_the_token_exemplar_names_no_build(self):
        exemplar = build_capabilities(
            capability_version=version._EXEMPLAR_CAPABILITY_VERSION, build=version._EXEMPLAR_BUILD
        )

        assert exemplar.data_builds == DataBuilds()
        assert all(value is None for value in exemplar.data_builds.model_dump().values())
