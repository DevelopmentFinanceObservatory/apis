"""The capabilities response publishes the constants enforced by this surface."""

from __future__ import annotations

from dfo_platform.build import build_commit
from graph_api.capabilities import Capabilities, build_capabilities
from graph_api.config import (
    GRAPH_NODE_POOL_SIZE,
    GRAPH_NODE_TIMEOUT_SECONDS,
    GRAPH_RESOLVE_TIMEOUT_SECONDS,
)
from graph_api.node_models import NODES_MAX_DCIDS
from graph_api.routing import DCID_MAX_LENGTH
from graph_api.version import GRAPH_CAPABILITY_VERSION

from .conftest import make_client


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
        assert doors == {"resolve": True, "node": True, "sources": True}

    def test_node_shape_gaps_match_the_recorded_permanent_null_fields(self):
        gaps = _capabilities().node_shape_gaps
        assert gaps.facets is False
        assert gaps.sources is False
        assert gaps.processing_level is False

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
